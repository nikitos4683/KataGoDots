#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/fancymath.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/parallel.h"
#include "../core/timer.h"
#include "../core/test.h"
#include "../dataio/sgf.h"
#include "../dataio/poswriter.h"
#include "../dataio/files.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../tests/tests.h"
#include "../neuralnet/sparsedotsnn.h"
#include "../main.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <mutex>
#include <thread>

using namespace std;

static std::atomic<bool> sigReceived(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM)
    sigReceived.store(true);
}

int MainCmds::printclockinfo(const vector<string>& args) {
  (void)args;
#ifdef OS_IS_WINDOWS
  cout << "Does nothing on windows, disabled" << endl;
#endif
#ifdef OS_IS_UNIX_OR_APPLE
  cout << "Tick unit in seconds: " << std::chrono::steady_clock::period::num << " / " <<  std::chrono::steady_clock::period::den << endl;
  cout << "Ticks since epoch: " << std::chrono::steady_clock::now().time_since_epoch().count() << endl;
#endif
  return 0;
}

int MainCmds::sampleinitializations(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  int numToGen;
  bool evaluate;
  try {
    KataGoCommandLine cmd("View startposes");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<int> numToGenArg("","num","Num to gen",false,1,"N");
    TCLAP::SwitchArg evaluateArg("","evaluate","Print out values and scores on the inited poses");
    cmd.add(numToGenArg);
    cmd.add(evaluateArg);
    cmd.parseArgs(args);
    numToGen = numToGenArg.getValue();
    evaluate = evaluateArg.getValue();

    cmd.getConfigAllowEmpty(cfg);
    if(cfg.getFileName() != "")
      modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  NNEvaluator* nnEval = NULL;
  if(cfg.getFileName() != "") {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    {
      Setup::initializeSession(cfg);
      const int expectedConcurrentEvals = params.numThreads;
      const bool defaultRequireExactNNLen = false;
      const bool disableFP16 = false;
      const string expectedSha256 = "";
      nnEval = Setup::initializeNNEvaluator(
        modelFile,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
        Board::MAX_LEN_X,Board::MAX_LEN_Y,Setup::MaxBatchSizeRequest::fromConcurrency(),defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_GTP
      );
    }
    logger.write("Loaded neural net");
  }

  AsyncBot* evalBot;
  {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);
    params.maxVisits = 20;
    params.numThreads = 1;
    string seed = Global::uint64ToString(rand.nextUInt64());
    evalBot = new AsyncBot(params, nnEval, &logger, seed);
  }

  //Play no moves in game, since we're sampling initializations
  cfg.overrideKey("maxMovesPerGame","0");

  const bool isDistributed = false;
  PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, isDistributed);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);

  for(int i = 0; i<numToGen; i++) {
    string seed = Global::uint64ToString(rand.nextUInt64());
    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = "";
    botSpec.nnEval = nnEval;
    botSpec.baseParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);

    FinishedGameData* data = gameRunner->runGame(
      seed,
      botSpec,
      botSpec,
      NULL,
      NULL,
      logger,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr
    );

    cout << data->startHist.rules.toString() << endl;
    Board::printBoard(cout, data->startBoard, Board::NULL_LOC, &(data->startHist.moveHistory));
    cout << endl;
    if(evaluate) {
      evalBot->setPosition(data->startPla, data->startBoard, data->startHist);
      evalBot->genMoveSynchronous(data->startPla,TimeControls());
      ReportedSearchValues values = evalBot->getSearchStopAndWait()->getRootValuesRequireSuccess();
      cout << "Winloss: " << values.winLossValue << endl;
      cout << "Lead: " << values.lead << endl;
    }

    delete data;
  }

  delete gameRunner;
  delete evalBot;
  if(nnEval != NULL)
    delete nnEval;

  ScoreValue::freeTables();
  return 0;
}

int MainCmds::evalrandominits(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  try {
    KataGoCommandLine cmd("Eval random inits");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    cmd.getConfigAllowEmpty(cfg);
    if(cfg.getFileName() != "")
      modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  NNEvaluator* nnEval = NULL;
  if(cfg.getFileName() != "") {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    {
      Setup::initializeSession(cfg);
      const int expectedConcurrentEvals = params.numThreads;
      const bool defaultRequireExactNNLen = false;
      const bool disableFP16 = false;
      const string expectedSha256 = "";
      nnEval = Setup::initializeNNEvaluator(
        modelFile,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
        Board::MAX_LEN_X,Board::MAX_LEN_Y,Setup::MaxBatchSizeRequest::fromConcurrency(),defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_GTP
      );
    }
    logger.write("Loaded neural net");
  }

  Search* evalBot;
  {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);
    params.maxVisits = 40;
    params.numThreads = 1;
    string seed = Global::uint64ToString(rand.nextUInt64());
    evalBot = new Search(params, nnEval, &logger, seed);
  }

  Rand gameRand;
  while(true) {
    Rules rules = Rules::parseRules("japanese");
    Board board(19,19,rules);
    Player pla = P_BLACK;
    //Keep pass-alive computations (e.g. endGameIfAllPassAlive below) consistent with how the bot's
    //own searches are performing them.
    BoardHistory hist(board,pla,rules,0,evalBot->getRootHist().modes);
    int numInitialMovesToPlay = (int)gameRand.nextUInt(200);
    double temperature = 1.0;
    for(int i = 0; i<numInitialMovesToPlay; i++) {
      NNResultBuf buf;
      Loc loc = PlayUtils::getGameInitializationMove(evalBot, evalBot, board, hist, pla, buf, gameRand, temperature);

      assert(hist.isLegal(board,loc,pla));
      hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
      pla = getOpp(pla);

      if (hist.endGameIfReasonable(board, true, pla))
        break;
    }

    evalBot->setPosition(pla,board,hist);
    evalBot->runWholeSearch(pla);
    ReportedSearchValues values = evalBot->getRootValuesRequireSuccess();
    cout << numInitialMovesToPlay << "," << values.winLossValue << "," << values.lead << endl;
  }
  delete evalBot;
  if(nnEval != NULL)
    delete nnEval;

  ScoreValue::freeTables();
  return 0;
}

int MainCmds::searchentropyanalysis(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  string boardSizeDataset;
  try {
    KataGoCommandLine cmd("Analyze search entropy across datasets");
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> boardSizeDatasetArg("","boardsizedataset", "Dataset to analyze (9,13,19,10x14,rectangle)", true, "", "SIZE");
    cmd.add(boardSizeDatasetArg);
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    boardSizeDataset = boardSizeDatasetArg.getValue();
    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  logger.write("Search Entropy Analysis");
  logger.write("Model: " + modelFile);
  logger.write("Dataset: " + boardSizeDataset);

  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
  NNEvaluator* nnEval = NULL;
  {
    Setup::initializeSession(cfg);
    const int expectedConcurrentEvals = params.numThreads;
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
      Board::MAX_LEN_X,Board::MAX_LEN_Y,Setup::MaxBatchSizeRequest::fromConcurrency(),defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_GTP
    );
  }
  logger.write("Loaded neural net");

  Search* bot;
  {
    SearchParams searchParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    string seed = Global::uint64ToString(rand.nextUInt64());
    bot = new Search(searchParams, nnEval, &logger, seed);
  }

  vector<string> sgfData;
  if(boardSizeDataset == "9")
    sgfData = TestCommon::getMultiGameSize9Data();
  else if(boardSizeDataset == "13")
    sgfData = TestCommon::getMultiGameSize13Data();
  else if(boardSizeDataset == "19")
    sgfData = TestCommon::getMultiGameSize19Data();
  else if(boardSizeDataset == "10x14")
    sgfData = TestCommon::getMultiGameSize10x14Data();
  else if(boardSizeDataset == "rectangle")
    sgfData = TestCommon::getMultiGameRectangleData();
  else {
    throw StringError("Unknown dataset to test gpu error on: " + boardSizeDataset);
  }

  // Build list of all (sgfIdx, turnIdx) pairs, then deterministically shuffle
  vector<std::pair<int,int>> positions;
  {
    vector<std::unique_ptr<CompactSgf>> sgfObjs;
    for(const string& sgf : sgfData)
      sgfObjs.push_back(CompactSgf::parse(sgf));
    for(int sgfIdx = 0; sgfIdx < (int)sgfObjs.size(); sgfIdx++) {
      for(int turnIdx = 0; turnIdx < (int)sgfObjs[sgfIdx]->moves.size(); turnIdx++) {
        positions.push_back({sgfIdx, turnIdx});
      }
    }
  }
  {
    Rand shuffleRand("searchentropyanalysis_shuffle_seed");
    shuffleRand.shuffle(positions);
  }
  logger.write("Total positions to search: " + Global::intToString((int)positions.size()));

  vector<double> searchEntropies;
  vector<double> searchSurprises;
  int numPositions = 0;

  auto printRunningStats = [&]() {
    cout << "Searched numPositions: " << numPositions << endl;
    if(numPositions > 0) {
      double entropyMean = 0.0;
      for(double v : searchEntropies) entropyMean += v;
      entropyMean /= numPositions;
      double entropyVar = 0.0;
      for(double v : searchEntropies) { double d = v - entropyMean; entropyVar += d * d; }
      entropyVar /= numPositions;
      cout << "Mean search entropy: " << entropyMean << " standard deviation: " << sqrt(entropyVar) << endl;

      double surpriseMean = 0.0;
      for(double v : searchSurprises) surpriseMean += v;
      surpriseMean /= numPositions;
      double surpriseVar = 0.0;
      for(double v : searchSurprises) { double d = v - surpriseMean; surpriseVar += d * d; }
      surpriseVar /= numPositions;
      cout << "Mean search surprise: " << surpriseMean << " standard deviation: " << sqrt(surpriseVar) << endl;
    }
  };

  for(const auto& pos : positions) {
    int sgfIdx = pos.first;
    int turnIdx = pos.second;

    std::unique_ptr<CompactSgf> sgfObj = CompactSgf::parse(sgfData[sgfIdx]);

    Player pla;
    Rules rules = sgfObj->getRulesOrFailAllowUnspecified(Rules::getSimpleTerritory());
    BoardHistory hist = sgfObj->setupInitialBoardAndHist(rules, pla, bot->getRootHist().modes);
    Board board = hist.initialBoard;

    for(int i = 0; i < turnIdx; i++) {
      Loc moveLoc = sgfObj->moves[i].loc;
      if(moveLoc != Board::NULL_LOC && hist.isLegal(board, moveLoc, pla)) {
        hist.makeBoardMoveAssumeLegal(board, moveLoc, pla, NULL);
        pla = getOpp(pla);
      }
    }

    if(!hist.isGameFinished) {
      bot->setPosition(pla, board, hist);
      bot->runWholeSearch(pla);

      double surprise, searchEntropy, policyEntropy;
      if(bot->getPolicySurpriseAndEntropy(surprise, searchEntropy, policyEntropy)) {
        searchEntropies.push_back(searchEntropy);
        searchSurprises.push_back(surprise);
        numPositions++;
        if(numPositions % 10 == 0)
          printRunningStats();
      }
    }
  }

  cout << modelFile << endl;
  cout << "Dataset: " << boardSizeDataset << endl;
  printRunningStats();

  delete bot;
  delete nnEval;
  ScoreValue::freeTables();
  return 0;
}

//One CSV row per recorded turn of the game, containing the stats that drive surprise-based selection of
//positions (policy surprise, value surprise) along with the per-game and per-turn covariates needed to
//analyze what they vary with.
static void writeSurpriseStatsRows(ostream& out, int64_t gameIdx, const FinishedGameData& data) {
  size_t numMoves = data.targetWeightByTurn.size();
  testAssert(data.hasFullData);
  testAssert(data.policySurpriseByTurn.size() == numMoves);
  testAssert(data.policyEntropyByTurn.size() == numMoves);
  testAssert(data.searchEntropyByTurn.size() == numMoves);
  testAssert(data.valueSurpriseByTurn.size() == numMoves);
  testAssert(data.wasCheapSearchByTurn.size() == numMoves);
  testAssert(data.nnRawStatsByTurn.size() == numMoves);
  testAssert(data.policyTargetsByTurn.size() == numMoves);
  testAssert(data.reanalysisByTurn.size() == numMoves);
  testAssert(data.whiteValueTargetsByTurn.size() == numMoves + 1);

  //Last entry of the value targets is the actual game outcome.
  const ValueTargets& outcome = data.whiteValueTargetsByTurn[numMoves];
  double pdaWhite =
    data.playoutDoublingAdvantagePla == P_WHITE ? data.playoutDoublingAdvantage :
    data.playoutDoublingAdvantagePla == P_BLACK ? -data.playoutDoublingAdvantage : 0.0;
  //Turn number of the first recorded turn, counting handicap/policy-init/fork setup moves and any initial
  //turn number offset of positions begun from sgfs, so that turn number is comparable across game modes.
  int64_t startTurnIdx = data.startHist.initialTurnNumber + (int64_t)data.startHist.moveHistory.size();

  for(size_t t = 0; t<numMoves; t++) {
    const ValueTargets& mcts = data.whiteValueTargetsByTurn[t];
    const NNRawStats& raw = data.nnRawStatsByTurn[t];
    const ReanalysisData& rean = data.reanalysisByTurn[t];
    out << gameIdx
        << "," << data.gameHash.toString()
        << "," << data.startBoard.x_size
        << "," << data.startBoard.y_size
        << "," << data.startHist.rules.komi
        << "," << pdaWhite
        << "," << data.numExtraBlack
        << "," << data.mode
        << "," << (data.hitTurnLimit ? 1 : 0)
        << "," << numMoves
        << "," << outcome.win
        << "," << outcome.loss
        << "," << outcome.noResult
        << "," << outcome.score
        << "," << (startTurnIdx + (int64_t)t)
        << "," << t
        << "," << (data.wasCheapSearchByTurn[t] ? 1 : 0)
        << "," << data.policyTargetsByTurn[t].unreducedNumVisits
        << "," << data.targetWeightByTurn[t]
        << "," << data.policySurpriseByTurn[t]
        << "," << data.valueSurpriseByTurn[t]
        << "," << data.policyEntropyByTurn[t]
        << "," << data.searchEntropyByTurn[t]
        << "," << raw.whiteWinLoss
        << "," << raw.whiteScoreMean
        << "," << mcts.win
        << "," << mcts.loss
        << "," << mcts.noResult
        << "," << mcts.score
        << "," << (rean.wasReanalyzed ? 1 : 0)
        << "," << rean.selectionPolicySurprise
        << "," << rean.selectionValueSurprise
        << "," << rean.originalNumVisits
        << "\n";
  }
}

int MainCmds::selfplaysurprisedump(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  string outputFile;
  int64_t numGamesTotal;
  bool reanalyzeAll;
  try {
    KataGoCommandLine cmd("Run selfplay games with a fixed model and dump per-position policy and value surprise stats to csv");
    cmd.addConfigFileArg("","selfplay config file");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> outputArg("","output","Csv file to write one row per recorded turn to",true,string(),"FILE");
    TCLAP::ValueArg<int> numGamesArg("","num-games","Number of games to run",true,0,"N");
    TCLAP::SwitchArg reanalyzeAllArg("","reanalyze-all","Instead of disabling reanalysis, reanalyze every cheap-search position, so each such row records both the cheap search's (before) and the full search's (after) surprise stats");
    cmd.add(outputArg);
    cmd.add(numGamesArg);
    cmd.add(reanalyzeAllArg);
    cmd.parseArgs(args);
    outputFile = outputArg.getValue();
    numGamesTotal = numGamesArg.getValue();
    reanalyzeAll = reanalyzeAllArg.getValue();
    if(numGamesTotal <= 0)
      throw StringError("-num-games must be positive");

    cmd.getConfig(cfg);
    modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand seedRand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  logger.write("Selfplay surprise stats dump");
  logger.write("Model: " + modelFile);

  const int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  const SearchParams baseParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_OTHER);
  const bool isDistributed = false;
  PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, isDistributed);
  if(reanalyzeAll) {
    //Reanalyze every cheap-search position, so that every cheap-search row records both the cheap search's
    //surprise stats (the ones that would drive reanalysis selection, preserved as origPolicySurprise and
    //origValueSurprise) and the full search's stats, without any selection bias in which positions get both.
    //Note that the post-reanalysis value surprise of a turn also absorbs the value changes of other reanalyzed
    //turns after it, via the smoothed forward-outcome track, so unlike policy surprise it is not a pure
    //cheap-vs-full comparison of this turn's search alone.
    logger.write("Reanalyzing all cheap-search positions to record before and after surprise stats");
    playSettings.useReanalyze = true;
    playSettings.reanalyzeProp = 1.0;
  }
  else {
    //Force no reanalysis, so that every turn's recorded stats are those of the search the game actually played
    //with - exactly the stats that would drive reanalysis selection - rather than partly overwritten by
    //reanalysis searches.
    if(playSettings.useReanalyze && playSettings.reanalyzeProp > 0.0)
      logger.write("Config has reanalyzeProp > 0, forcing it to 0 so that recorded stats are the pre-reanalysis ones");
    playSettings.reanalyzeProp = 0.0;
  }

  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);

  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();

  Setup::initializeSession(cfg);
  NNEvaluator* nnEval;
  {
    const int expectedConcurrentEvals = cfg.getInt("numSearchThreads") * numGameThreads;
    const bool defaultRequireExactNNLen = minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      maxBoardXSizeUsed,maxBoardYSizeUsed,Setup::MaxBatchSizeRequest::requireFromConfig(),defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_OTHER
    );
  }
  logger.write("Loaded neural net");

  cfg.warnUnusedKeys(cerr,&logger);

  ofstream out;
  FileUtils::open(out,outputFile);
  out.precision(9);
  out << "gameIdx,gameHash,xSize,ySize,komi,pdaWhite,numExtraBlack,mode,hitTurnLimit,gameNumMoves"
      << ",outcomeWhiteWin,outcomeWhiteLoss,outcomeWhiteNoResult,outcomeWhiteScore"
      << ",turnIdx,turnAfterStart,wasCheapSearch,unreducedNumVisits,targetWeight"
      << ",policySurprise,valueSurprise,policyEntropy,searchEntropy"
      << ",rawNNWhiteWinLoss,rawNNWhiteScoreMean,mctsWhiteWin,mctsWhiteLoss,mctsWhiteNoResult,mctsWhiteScore"
      << ",wasReanalyzed,origPolicySurprise,origValueSurprise,origNumVisits"
      << "\n";

  if(!std::atomic_is_lock_free(&sigReceived))
    throw StringError("sigReceived is not lock free, signal-quitting mechanism will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  std::mutex writeMutex;
  std::atomic<int64_t> numGamesStarted(0);
  int64_t numGamesWritten = 0;
  ForkData* forkData = new ForkData();

  auto gameLoop = [
    &gameRunner,&logger,&numGamesStarted,&forkData,numGamesTotal,&baseParams,&gameSeedBase,
    &nnEval,&writeMutex,&out,&numGamesWritten
  ](int threadIdx) {
    (void)threadIdx;
    auto shouldStopFunc = []() noexcept {
      return sigReceived.load();
    };
    Rand thisLoopSeedRand;
    while(true) {
      if(sigReceived.load())
        break;
      int64_t gameIdx = numGamesStarted.fetch_add(1,std::memory_order_acq_rel);
      if(gameIdx >= numGamesTotal)
        break;

      MatchPairer::BotSpec botSpec;
      botSpec.botIdx = 0;
      botSpec.botName = nnEval->getModelName();
      botSpec.nnEval = nnEval;
      botSpec.baseParams = baseParams;

      string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
      FinishedGameData* gameData = gameRunner->runGame(
        seed, botSpec, botSpec, forkData, NULL, logger,
        shouldStopFunc, nullptr, nullptr, nullptr, nullptr
      );
      //Happens when interrupted by the signal handler mid-game
      if(gameData == NULL)
        break;

      {
        std::lock_guard<std::mutex> lock(writeMutex);
        writeSurpriseStatsRows(out, gameIdx, *gameData);
        out.flush();
        numGamesWritten += 1;
        logger.write(
          "Finished game " + Global::int64ToString(gameIdx) + " (" +
          Global::int64ToString(numGamesWritten) + "/" + Global::int64ToString(numGamesTotal) + " written)"
        );
      }
      delete gameData;
    }
  };

  vector<std::thread> threads;
  for(int i = 0; i<numGameThreads; i++)
    threads.push_back(std::thread(gameLoop,i));
  for(int i = 0; i<numGameThreads; i++)
    threads[i].join();

  out.close();
  logger.write("Wrote stats for " + Global::int64ToString(numGamesWritten) + " games to " + outputFile);

  delete forkData;
  delete gameRunner;
  delete nnEval;
  ScoreValue::freeTables();
  return 0;
}

int MainCmds::dumpsparse(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  string sgfFile;
  int moveNum;
  int symmetry;
  bool printBoard;
  int maxHistory;
  try {
    KataGoCommandLine cmd("Print the sparse Dots NN input (model version 18) of a position from an sgf file as JSON.");
    TCLAP::UnlabeledValueArg<string> sgfFileArg("","Sgf file",true,string(),"FILE");
    TCLAP::ValueArg<int> moveNumArg("m","move-num","Sgf move num, the end of the game by default",false,-1,"MOVENUM");
    TCLAP::ValueArg<int> symmetryArg("","symmetry","Symmetry (0-7) applied to the coordinates",false,0,"SYMMETRY");
    TCLAP::ValueArg<int> maxHistoryArg("","max-history","Max number of moves of history",false,1000,"N");
    TCLAP::SwitchArg printBoardArg("","print-board","Also print the board before the JSON");
    cmd.add(sgfFileArg);
    cmd.add(moveNumArg);
    cmd.add(symmetryArg);
    cmd.add(maxHistoryArg);
    cmd.add(printBoardArg);
    cmd.parseArgs(args);
    sgfFile = sgfFileArg.getValue();
    moveNum = moveNumArg.getValue();
    symmetry = symmetryArg.getValue();
    maxHistory = maxHistoryArg.getValue();
    printBoard = printBoardArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  const std::unique_ptr<CompactSgf> sgf = CompactSgf::loadFile(sgfFile);
  if(!sgf->isDots)
    throw StringError("dumpsparse: the sgf is not a Dots game");
  Rules rules = sgf->getRulesOrWarn(Rules::DEFAULT_DOTS, [](const string& msg) { cerr << msg << endl; });
  // Tolerate moves into the opponent's houses that real games may contain (5.14)
  rules.multiStoneSuicideLegal = true;
  if(moveNum < 0)
    moveNum = static_cast<int>(sgf->moves.size());
  if(moveNum > sgf->moves.size())
    throw StringError("Move num " + Global::intToString(moveNum) + " requested but sgf has only " + Global::int64ToString(sgf->moves.size()));
  if(symmetry < 0 || symmetry >= SymmetryHelpers::NUM_SYMMETRIES)
    throw StringError("Invalid symmetry " + Global::intToString(symmetry));

  Player nextPla;
  BoardHistory hist = sgf->setupInitialBoardAndHist(rules, nextPla, BoardHistoryModes());
  Board board = hist.initialBoard;
  sgf->playMovesTolerant(board, nextPla, hist, moveNum, false);
  if(SymmetryHelpers::isTranspose(symmetry) && board.x_size != board.y_size)
    throw StringError("Transposing symmetries are allowed only for square boards");

  MiscNNInputParams params;
  params.maxHistory = maxHistory;
  DotsSparseInput input;
  NNInputs::fillDotsSparseInput(board, hist, nextPla, params, input, false);
  if(printBoard) {
    Board::printBoard(cout, board, Board::NULL_LOC, &hist.moveHistory);
    cout << "Next player: " << PlayerIO::playerToString(nextPla, true) << endl;
  }
  cout << input.withSymmetry(symmetry).toJson().dump(1) << endl;
  return 0;
}

// Positions of random Dots games (random moves preferring captures), sampled over the whole games
static vector<DotsSparseInput> randomSparseDotsPositions(Rand& rand, int xSize, int ySize, int numPositions, int& numGames) {
  vector<DotsSparseInput> result;
  numGames = 0;
  while((int)result.size() < numPositions) {
    Rules rules = Rules::DEFAULT_DOTS;
    rules.startPos = rand.nextBool(0.5) ? Rules::START_POS_CROSS : Rules::START_POS_CROSS_4;
    rules.startPosIsRandom = rules.startPos == Rules::START_POS_CROSS_4;
    Board board(xSize, ySize, rules);
    BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
    Player pla = P_BLACK;
    numGames++;
    // Sample positions from the whole game, more of them later when there are more areas
    const double recordProb = 4.0 / (xSize * ySize * 0.3);
    for(int moveIdx = 0; moveIdx < xSize * ySize && !hist.isGameFinished && (int)result.size() < numPositions; moveIdx++) {
      MiscNNInputParams params;
      DotsSparseInput input;
      NNInputs::fillDotsSparseInput(board, hist, pla, params, input, false);
      if(input.candidates.empty() && !input.groundAllowed)
        break;
      if(rand.nextBool(recordProb))
        result.push_back(input);
      Loc loc;
      if(input.candidates.empty() || (input.groundAllowed && rand.nextBool(0.02))) {
        loc = Board::PASS_LOC;
      }
      else {
        vector<int> captures;
        for(int i = 0; i < (int)input.candidates.size(); i++)
          if(input.candidates[i].gainMy != 0)
            captures.push_back(i);
        int idx;
        if(!captures.empty() && rand.nextBool(0.5))
          idx = captures[rand.nextUInt((uint32_t)captures.size())];
        else
          idx = (int)rand.nextUInt((uint32_t)input.candidates.size());
        loc = input.candidateLocs[idx];
      }
      hist.makeBoardMoveAssumeLegal(board, loc, pla, nullptr);
      pla = getOpp(pla);
    }
  }
  return result;
}

int MainCmds::dumpsparsenn(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  string modelFile;
  string outputFile;
  int numPositions;
  int xSize;
  int ySize;
  string seed;
  try {
    KataGoCommandLine cmd(
      "Evaluate positions of random Dots games with the CPU reference of a sparse model (version 18) and write the "
      "inputs and the raw outputs as JSON, to compare with other implementations.");
    TCLAP::ValueArg<string> modelArg("","model","Sparse Dots model file",true,string(),"FILE");
    TCLAP::ValueArg<string> outputArg("","output","JSON file to write",true,string(),"FILE");
    TCLAP::ValueArg<int> numPositionsArg("","num-positions","Number of positions",false,40,"N");
    TCLAP::ValueArg<int> xSizeArg("","x-size","Board x size",false,Board::DEFAULT_LEN_X_DOTS,"X");
    TCLAP::ValueArg<int> ySizeArg("","y-size","Board y size",false,Board::DEFAULT_LEN_Y_DOTS,"Y");
    TCLAP::ValueArg<string> seedArg("","seed","Random seed",false,"dumpsparsenn","SEED");
    cmd.add(modelArg);
    cmd.add(outputArg);
    cmd.add(numPositionsArg);
    cmd.add(xSizeArg);
    cmd.add(ySizeArg);
    cmd.add(seedArg);
    cmd.parseArgs(args);
    modelFile = modelArg.getValue();
    outputFile = outputArg.getValue();
    numPositions = numPositionsArg.getValue();
    xSize = xSizeArg.getValue();
    ySize = ySizeArg.getValue();
    seed = seedArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  ModelDesc desc;
  ModelDesc::loadFromFileMaybeGZipped(modelFile, desc, "");
  if(!desc.isSparseDots())
    throw StringError("dumpsparsenn: not a sparse Dots model");
  const SparseDotsNN::Model model(*desc.sparseDots);

  Rand rand(seed);
  int numGames = 0;
  const vector<DotsSparseInput> positions = randomSparseDotsPositions(rand, xSize, ySize, numPositions, numGames);
  nlohmann::json result = nlohmann::json::array();
  for(const DotsSparseInput& input : positions) {
    const int symmetry = SparseDotsNN::effectiveSymmetry(
      (int)rand.nextUInt(SymmetryHelpers::NUM_SYMMETRIES), input.xSize, input.ySize);
    const DotsSparseInput transformed = input.withSymmetry(symmetry);
    SparseDotsNN::RawOutput raw;
    model.evaluate(transformed, raw);
    nlohmann::json entry;
    entry["symmetry"] = symmetry;
    entry["input"] = transformed.toJson();
    entry["policy"] = raw.policy;
    entry["value"] = std::vector<float>(raw.value, raw.value + 3);
    entry["scoreValue"] = std::vector<float>(raw.scoreValue, raw.scoreValue + 6);
    entry["ownership"] = raw.ownership;
    result.push_back(entry);
  }
  ofstream out;
  FileUtils::open(out, outputFile);
  out << result.dump() << endl;
  out.close();
  cout << "Wrote " << result.size() << " positions from " << numGames << " games to " << outputFile << endl;
  return 0;
}

int MainCmds::testsparsenn(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  string modelFile;
  int numPositions;
  int batchSize;
  int xSize;
  int ySize;
  string seed;
  string useFP16Str;
  double tolerance;
  try {
    KataGoCommandLine cmd(
      "Evaluate positions of random Dots games with the neural net backend of this build and with the CPU reference "
      "of a sparse model (version 18), and compare the raw outputs.");
    TCLAP::ValueArg<string> modelArg("","model","Sparse Dots model file",true,string(),"FILE");
    TCLAP::ValueArg<int> numPositionsArg("","num-positions","Number of positions",false,64,"N");
    TCLAP::ValueArg<int> batchSizeArg("","batch-size","Max batch size, the batches have random sizes up to it",false,8,"N");
    TCLAP::ValueArg<int> xSizeArg("","x-size","Board x size",false,Board::DEFAULT_LEN_X_DOTS,"X");
    TCLAP::ValueArg<int> ySizeArg("","y-size","Board y size",false,Board::DEFAULT_LEN_Y_DOTS,"Y");
    TCLAP::ValueArg<string> seedArg("","seed","Random seed",false,"testsparsenn","SEED");
    TCLAP::ValueArg<string> useFP16Arg("","use-fp16","true, false or auto",false,"false","BOOL");
    TCLAP::ValueArg<double> toleranceArg("","tolerance","Max allowed absolute difference of any output, 0 = default "
                                         "(0.002 in FP32, 0.05 in FP16)",false,0.0,"X");
    cmd.add(modelArg);
    cmd.add(numPositionsArg);
    cmd.add(batchSizeArg);
    cmd.add(xSizeArg);
    cmd.add(ySizeArg);
    cmd.add(seedArg);
    cmd.add(useFP16Arg);
    cmd.add(toleranceArg);
    cmd.parseArgs(args);
    modelFile = modelArg.getValue();
    numPositions = numPositionsArg.getValue();
    batchSize = batchSizeArg.getValue();
    xSize = xSizeArg.getValue();
    ySize = ySizeArg.getValue();
    seed = seedArg.getValue();
    useFP16Str = useFP16Arg.getValue();
    tolerance = toleranceArg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  enabled_t useFP16;
  if(!enabled_t::tryParse(useFP16Str, useFP16))
    throw StringError("testsparsenn: bad use-fp16 value " + useFP16Str);
  if(batchSize < 1)
    throw StringError("testsparsenn: batch-size must be positive");

  Logger logger(nullptr, true);
  NeuralNet::globalInitialize();
  LoadedModel* loadedModel = NeuralNet::loadModelFile(modelFile, "");
  const ModelDesc& desc = NeuralNet::getModelDesc(loadedModel);
  if(!desc.isSparseDots())
    throw StringError("testsparsenn: not a sparse Dots model");
  const SparseDotsNN::Model reference(*desc.sparseDots);

  // Transposing symmetries are not applied to rectangular boards, so the outputs have the board shape
  const int nnXLen = xSize;
  const int nnYLen = ySize;
  if(nnXLen < 1 || nnYLen < 1 || nnXLen > NNPos::MAX_BOARD_LEN_X || nnYLen > NNPos::MAX_BOARD_LEN_Y)
    throw StringError("testsparsenn: the board size doesn't fit the compiled max board size");
  ConfigParser cfg(std::map<string, string>{});
  ComputeContext* context = NeuralNet::createComputeContext({0}, &logger, nnXLen, nnYLen, "", useFP16, loadedModel, cfg);
  ComputeHandle* handle = NeuralNet::createComputeHandle(context, loadedModel, &logger, batchSize, false, true, 0, 0);
  InputBuffers* inputBuffers = NeuralNet::createInputBuffers(loadedModel, batchSize, nnXLen, nnYLen);
  const bool usingFP16 = NeuralNet::isUsingFP16(handle);
  if(tolerance <= 0.0)
    tolerance = usingFP16 ? 0.05 : 0.002;

  Rand rand(seed);
  int numGames = 0;
  const vector<DotsSparseInput> positions = randomSparseDotsPositions(rand, xSize, ySize, numPositions, numGames);
  cout << "Positions: " << positions.size() << " from " << numGames << " games, FP16 " << Global::boolToString(usingFP16)
       << ", tolerance " << tolerance << endl;

  auto newOutput = [&]() {
    auto output = std::make_unique<NNOutput>();
    output->nnXLen = nnXLen;
    output->nnYLen = nnYLen;
    output->whiteOwnerMap = new float[nnXLen * nnYLen];
    output->noisedPolicyProbs = nullptr;
    return output;
  };
  double maxPolicyDiff = 0.0, maxValueDiff = 0.0, maxScoreDiff = 0.0, maxOwnershipDiff = 0.0;
  double sumPolicyDiff = 0.0;
  int64_t numPolicyValues = 0;
  int numBatches = 0;
  size_t start = 0;
  while(start < positions.size()) {
    const int n = std::min<int>((int)(positions.size() - start), 1 + (int)rand.nextUInt((uint32_t)batchSize));
    vector<std::unique_ptr<NNResultBuf>> bufs;
    vector<NNResultBuf*> bufPtrs;
    vector<std::unique_ptr<NNOutput>> outputs;
    vector<NNOutput*> outputPtrs;
    for(int i = 0; i < n; i++) {
      bufs.push_back(std::make_unique<NNResultBuf>());
      NNResultBuf& buf = *bufs.back();
      buf.dotsSparseInput = positions[start + i];
      buf.symmetry = (int)rand.nextUInt(SymmetryHelpers::NUM_SYMMETRIES);
      buf.policyOptimism = rand.nextBool(0.5) ? 0.0 : 1.0;
      bufPtrs.push_back(&buf);
      outputs.push_back(newOutput());
      outputPtrs.push_back(outputs.back().get());
    }
    NeuralNet::getOutput(handle, inputBuffers, n, bufPtrs.data(), outputPtrs, true);

    for(int i = 0; i < n; i++) {
      const DotsSparseInput& input = positions[start + i];
      const int symmetry = SparseDotsNN::effectiveSymmetry(bufs[i]->symmetry, input.xSize, input.ySize);
      SparseDotsNN::RawOutput raw;
      reference.evaluate(input.withSymmetry(symmetry), raw);
      auto expected = newOutput();
      SparseDotsNN::writeOutput(input, raw, (float)bufs[i]->policyOptimism, *expected);
      const NNOutput& actual = *outputs[i];

      vector<int> policyPositions;
      for(Loc loc : input.candidateLocs)
        policyPositions.push_back(NNPos::locToPos(loc, input.xSize, nnXLen, nnYLen));
      if(input.groundAllowed)
        policyPositions.push_back(NNPos::getPassPos(nnXLen, nnYLen));
      for(int pos : policyPositions) {
        const double diff = std::fabs((double)actual.policyProbs[pos] - expected->policyProbs[pos]);
        if(diff > tolerance && diff > maxPolicyDiff)
          cout << "Policy diff " << diff << " at position " << (start + i) << " pos " << pos << " (pass "
               << NNPos::getPassPos(nnXLen, nnYLen) << "): " << actual.policyProbs[pos] << " vs " << expected->policyProbs[pos]
               << ", candidates " << input.candidates.size() << ", ground " << input.groundAllowed << ", points "
               << input.points.size() << ", areas " << input.areas.size() << endl;
        maxPolicyDiff = std::max(maxPolicyDiff, diff);
        sumPolicyDiff += diff;
        numPolicyValues++;
      }
      const float actualValues[3] = {actual.whiteWinProb, actual.whiteLossProb, actual.whiteNoResultProb};
      const float expectedValues[3] = {expected->whiteWinProb, expected->whiteLossProb, expected->whiteNoResultProb};
      for(int j = 0; j < 3; j++)
        maxValueDiff = std::max(maxValueDiff, std::fabs((double)actualValues[j] - expectedValues[j]));
      const float actualScores[6] = {actual.whiteScoreMean, actual.whiteScoreMeanSq, actual.whiteLead, actual.varTimeLeft,
                                     actual.shorttermWinlossError, actual.shorttermScoreError};
      const float expectedScores[6] = {expected->whiteScoreMean, expected->whiteScoreMeanSq, expected->whiteLead,
                                       expected->varTimeLeft, expected->shorttermWinlossError, expected->shorttermScoreError};
      for(int j = 0; j < 6; j++)
        maxScoreDiff = std::max(maxScoreDiff, std::fabs((double)actualScores[j] - expectedScores[j]));
      for(Loc loc : input.pointLocs) {
        const int pos = NNPos::locToPos(loc, input.xSize, nnXLen, nnYLen);
        maxOwnershipDiff = std::max(maxOwnershipDiff, std::fabs((double)actual.whiteOwnerMap[pos] - expected->whiteOwnerMap[pos]));
      }
    }
    start += n;
    numBatches++;
  }

  cout << "Batches: " << numBatches << endl;
  cout << "Max abs diff: policy " << maxPolicyDiff << " (mean " << (sumPolicyDiff / std::max<int64_t>(1, numPolicyValues))
       << "), value " << maxValueDiff << ", score value " << maxScoreDiff << ", ownership " << maxOwnershipDiff << endl;
  const bool ok = maxPolicyDiff <= tolerance && maxValueDiff <= tolerance && maxScoreDiff <= tolerance &&
                  maxOwnershipDiff <= tolerance;
  cout << (ok ? "OK" : "FAILED") << endl;

  NeuralNet::freeInputBuffers(inputBuffers);
  NeuralNet::freeComputeHandle(handle);
  NeuralNet::freeComputeContext(context);
  NeuralNet::freeLoadedModel(loadedModel);
  NeuralNet::globalCleanup();
  return ok ? 0 : 1;
}

// Positions of the games of sgfs files, each kept with the given probability
static vector<DotsSparseInput> sgfsSparseDotsPositions(Rand& rand, const vector<string>& sgfsFiles, double keepProb, int& numGames) {
  vector<DotsSparseInput> result;
  numGames = 0;
  for(std::unique_ptr<Sgf>& sgfPtr : Sgf::loadSgfsFiles(sgfsFiles)) {
    const CompactSgf sgf(std::move(*sgfPtr));
    if(!sgf.isDots)
      continue;
    Rules rules = sgf.getRulesOrFail();
    // Tolerate moves into the opponent's houses that real games may contain (5.14)
    rules.multiStoneSuicideLegal = true;
    Player pla;
    BoardHistory hist = sgf.setupInitialBoardAndHist(rules, pla, BoardHistoryModes());
    Board board = hist.initialBoard;
    numGames++;
    for(const Move& move : sgf.moves) {
      if(hist.isGameFinished)
        break;
      if(rand.nextBool(keepProb)) {
        MiscNNInputParams params;
        DotsSparseInput input;
        NNInputs::fillDotsSparseInput(board, hist, pla, params, input, false);
        result.push_back(input);
      }
      if(!hist.isLegal(board, move.loc, move.pla))
        break;
      hist.makeBoardMoveAssumeLegal(board, move.loc, move.pla, nullptr);
      pla = getOpp(move.pla);
    }
  }
  return result;
}

int MainCmds::benchsparsenn(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  string modelFile;
  vector<string> sgfsFiles;
  int numPositions;
  int batchSize;
  int numBatches;
  int numThreads;
  string useFP16Str;
  try {
    KataGoCommandLine cmd(
      "Measure the speed of the neural net backend of this build on a sparse Dots model (version 18). The batches mix "
      "positions of different games and stages, as in selfplay.");
    TCLAP::ValueArg<string> modelArg("","model","Sparse Dots model file",true,string(),"FILE");
    TCLAP::MultiArg<string> sgfsArg("","sgfs","Sgfs files with the games to take the positions from (random games if none)",false,"FILE");
    TCLAP::ValueArg<int> numPositionsArg("","num-positions","Number of distinct positions",false,4096,"N");
    TCLAP::ValueArg<int> batchSizeArg("","batch-size","Batch size",false,64,"N");
    TCLAP::ValueArg<int> numBatchesArg("","num-batches","Number of timed batches per thread",false,100,"N");
    TCLAP::ValueArg<int> numThreadsArg("","threads","Number of threads, each with its own compute handle",false,1,"N");
    TCLAP::ValueArg<string> useFP16Arg("","use-fp16","true, false or auto",false,"auto","BOOL");
    cmd.add(modelArg);
    cmd.add(sgfsArg);
    cmd.add(numPositionsArg);
    cmd.add(batchSizeArg);
    cmd.add(numBatchesArg);
    cmd.add(numThreadsArg);
    cmd.add(useFP16Arg);
    cmd.parseArgs(args);
    modelFile = modelArg.getValue();
    sgfsFiles = sgfsArg.getValue();
    numPositions = numPositionsArg.getValue();
    batchSize = batchSizeArg.getValue();
    numBatches = numBatchesArg.getValue();
    numThreads = numThreadsArg.getValue();
    useFP16Str = useFP16Arg.getValue();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  enabled_t useFP16;
  if(!enabled_t::tryParse(useFP16Str, useFP16))
    throw StringError("benchsparsenn: bad use-fp16 value " + useFP16Str);
  if(batchSize < 1 || numBatches < 1 || numPositions < 1 || numThreads < 1)
    throw StringError("benchsparsenn: batch-size, num-batches, num-positions and threads must be positive");

  Rand rand("benchsparsenn");
  int numGames = 0;
  vector<DotsSparseInput> positions;
  if(sgfsFiles.empty()) {
    positions = randomSparseDotsPositions(rand, Board::DEFAULT_LEN_X_DOTS, Board::DEFAULT_LEN_Y_DOTS, numPositions, numGames);
  }
  else {
    // Keep about numPositions positions of all the games
    int numAll = 0;
    positions = sgfsSparseDotsPositions(rand, sgfsFiles, 1.0, numGames);
    numAll = (int)positions.size();
    if(numAll > numPositions) {
      for(int i = numAll - 1; i > 0; i--)
        std::swap(positions[i], positions[rand.nextUInt((uint32_t)(i + 1))]);
      positions.resize(numPositions);
    }
  }
  if(positions.empty())
    throw StringError("benchsparsenn: no positions");
  for(int i = (int)positions.size() - 1; i > 0; i--)
    std::swap(positions[i], positions[rand.nextUInt((uint32_t)(i + 1))]);
  const int xSize = positions[0].xSize;
  const int ySize = positions[0].ySize;
  double sumPoints = 0.0, sumCands = 0.0;
  for(const DotsSparseInput& input : positions) {
    if(input.xSize != xSize || input.ySize != ySize)
      throw StringError("benchsparsenn: all the games must have the same board size");
    sumPoints += (double)input.points.size();
    sumCands += (double)input.candidates.size();
  }

  Logger logger(nullptr, true);
  NeuralNet::globalInitialize();
  LoadedModel* loadedModel = NeuralNet::loadModelFile(modelFile, "");
  if(!NeuralNet::getModelDesc(loadedModel).isSparseDots())
    throw StringError("benchsparsenn: not a sparse Dots model");
  ConfigParser cfg(std::map<string, string>{});
  ComputeContext* context = NeuralNet::createComputeContext({0}, &logger, xSize, ySize, "", useFP16, loadedModel, cfg);
  cout << "Positions: " << positions.size() << " from " << numGames << " games, " << xSize << "x" << ySize
       << ", mean points " << sumPoints / positions.size() << ", mean candidates " << sumCands / positions.size()
       << ", batch size " << batchSize << ", threads " << numThreads << endl;

  // Every thread has its own compute handle (and CUDA stream), like the NN server threads of an NNEvaluator
  std::atomic<int> numReady(0);
  std::atomic<bool> go(false);
  std::atomic<bool> usingFP16(false);
  auto worker = [&](int threadIdx) {
    ComputeHandle* handle = NeuralNet::createComputeHandle(context, loadedModel, &logger, batchSize, false, true, 0, threadIdx);
    InputBuffers* inputBuffers = NeuralNet::createInputBuffers(loadedModel, batchSize, xSize, ySize);
    usingFP16 = NeuralNet::isUsingFP16(handle);
    Rand threadRand("benchsparsenn" + Global::intToString(threadIdx));
    vector<NNResultBuf> bufs(batchSize);
    vector<NNResultBuf*> bufPtrs;
    vector<NNOutput> outputs(batchSize);
    vector<NNOutput*> outputPtrs;
    vector<vector<float>> ownerMaps(batchSize, vector<float>((size_t)xSize * ySize));
    for(int i = 0; i < batchSize; i++) {
      bufPtrs.push_back(&bufs[i]);
      outputs[i].nnXLen = xSize;
      outputs[i].nnYLen = ySize;
      outputs[i].whiteOwnerMap = ownerMaps[i].data();
      outputs[i].noisedPolicyProbs = nullptr;
      outputPtrs.push_back(&outputs[i]);
    }
    size_t next = (positions.size() * threadIdx) / numThreads;
    auto runBatch = [&]() {
      for(int i = 0; i < batchSize; i++) {
        bufs[i].dotsSparseInput = positions[next];
        bufs[i].symmetry = (int)threadRand.nextUInt(SymmetryHelpers::NUM_SYMMETRIES);
        bufs[i].policyOptimism = 0.0;
        next = (next + 1) % positions.size();
      }
      NeuralNet::getOutput(handle, inputBuffers, batchSize, bufPtrs.data(), outputPtrs, true);
    };
    const int numWarmup = std::max(5, numBatches / 10);
    for(int i = 0; i < numWarmup; i++)
      runBatch();
    numReady++;
    while(!go)
      std::this_thread::yield();
    for(int i = 0; i < numBatches; i++)
      runBatch();
    for(NNOutput& output : outputs)
      output.whiteOwnerMap = nullptr;
    NeuralNet::freeInputBuffers(inputBuffers);
    NeuralNet::freeComputeHandle(handle);
  };
  vector<std::thread> threads;
  for(int t = 0; t < numThreads; t++)
    threads.emplace_back(worker, t);
  while(numReady < numThreads)
    std::this_thread::yield();
  ClockTimer timer;
  go = true;
  for(std::thread& thread : threads)
    thread.join();
  const double seconds = timer.getSeconds();
  const double totalBatches = (double)numBatches * numThreads;
  cout << "FP16 " << Global::boolToString(usingFP16) << ", batches: " << (int64_t)totalBatches << ", "
       << (seconds * 1000.0 / numBatches) << " ms per batch of a thread, " << (totalBatches * batchSize / seconds)
       << " positions per second" << endl;

  NeuralNet::freeComputeContext(context);
  NeuralNet::freeLoadedModel(loadedModel);
  NeuralNet::globalCleanup();
  return 0;
}
