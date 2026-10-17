#include "../tests/tests.h"

#include <algorithm>
#include <fstream>

#include <ghc/filesystem.hpp>

#include "../core/config_parser.h"
#include "../core/fileutils.h"
#include "../dataio/sgf.h"
#include "../program/play.h"

using namespace std;
using namespace TestCommon;
namespace gfs = ghc::filesystem;

namespace {

struct DotsGame {
  BoardHistory hist;
  //The board after each move, the first one is the start pos
  vector<Board> boards;
};

//A random game ended by a grounding, as selfplay ends a game.
DotsGame playRandomDotsGame(const Rules& rules, const int xSize, const int ySize, const int numMoves, Rand& rand) {
  Board board(xSize, ySize, rules);
  Player pla = board.setStartPos(rand);
  BoardHistory hist(board, pla, rules, 0, BoardHistoryModes());
  hist.setInitialTurnNumber(rules.getNumOfStartPosStones());
  vector<Board> boards = {board};

  vector<Loc> locs;
  for(int y = 0; y < ySize; y++)
    for(int x = 0; x < xSize; x++)
      locs.push_back(Location::getLoc(x, y, xSize));
  rand.shuffle(locs);

  for(const Loc loc : locs) {
    if(static_cast<int>(hist.moveHistory.size()) >= numMoves)
      break;
    if(!hist.isLegal(board, loc, pla))
      continue;
    hist.makeBoardMoveAssumeLegal(board, loc, pla, nullptr);
    testAssert(!hist.isGameFinished);
    boards.push_back(board);
    pla = getOpp(pla);
  }
  hist.makeBoardMoveAssumeLegal(board, Board::PASS_LOC, pla, nullptr);
  testAssert(hist.isGameFinished);
  //Bases make a Dots position more than its dots, the test is pointless without them
  testAssert(board.numBlackCaptures + board.numWhiteCaptures > 0);
  return {hist, boards};
}

//Board::isEqualForTesting, except that the start pos dots may be in any order (an SGF orders them by recognition)
bool dotsBoardsEqual(const Board& board, const Board& other, const bool checkRules) {
  auto sortedStartPosMoves = [](const Board& b) {
    vector<pair<Loc, Player>> moves;
    for(const Move& move : b.start_pos_moves)
      moves.emplace_back(move.loc, move.pla);
    std::sort(moves.begin(), moves.end());
    return moves;
  };
  if(sortedStartPosMoves(board) != sortedStartPosMoves(other))
    return false;
  Board boardWithOtherStartPosMoves = board;
  boardWithOtherStartPosMoves.start_pos_moves = other.start_pos_moves;
  return boardWithOtherStartPosMoves.isEqualForTesting(other, true, true, checkRules);
}

string toSgfLine(const BoardHistory& hist) {
  ostringstream out;
  WriteSgf::writeSgf(out, "b", "w", hist, nullptr, false, false);
  return out.str();
}

void checkSamplesOfGame(const DotsGame& game) {
  const vector<Move>& gameMoves = game.hist.moveHistory;
  const Board& startBoard = game.boards[0];

  const std::unique_ptr<Sgf> sgf = Sgf::parse(toSgfLine(game.hist));
  std::set<Hash128> uniqueHashes;
  vector<Sgf::PositionSample> samples;
  sgf->loadAllUniquePositions(uniqueHashes, false, false, true, false, nullptr, samples);

  //The start pos and the position after each move but the grounding that ends the game
  testAssert(samples.size() == gameMoves.size());
  for(size_t i = 0; i < samples.size(); i++) {
    const Sgf::PositionSample& sample = samples[i];
    //A Dots sample is its game from the start pos, under the rules of the SGF
    testAssert(dotsBoardsEqual(sample.board, startBoard, true));
    testAssert(sample.initialTurnNumber == game.hist.initialTurnNumber);
    testAssert(sample.nextPla == game.hist.initialPla);
    testAssert(sample.moves.size() == i);
    for(size_t j = 0; j < i; j++)
      testAssert(movesEqual(sample.moves[j], gameMoves[j]));

    Player nextPla;
    BoardHistory hist;
    testAssert(sample.tryGetCurrentBoardHistory(game.hist.rules, nextPla, hist, BoardHistoryModes()));
    testAssert(dotsBoardsEqual(hist.getRecentBoard(0), game.boards[i], true));
    testAssert(nextPla == gameMoves[i].pla);
  }
}

const string& getConfigDotsKeys() {
  static const string keys =
    "dots = true\n"
    "dotsCaptureEmptyBases = true\n"
    "suicideLegals = false\n"
    "bSizesXY = 39-32\n"
    "bSizeRelProbs = 1\n"
    "komiMean = 0\n"
    "startPosesLoadProb = 1.0\n"
    "startPosesTurnWeightLambda = 0\n";
  return keys;
}

std::unique_ptr<GameInitializer> createGameInitializer(const string& cfgStr, ostream* log = nullptr) {
  istringstream in(getConfigDotsKeys() + cfgStr);
  ConfigParser cfg(in);
  Logger logger(nullptr, false, false, false, false);
  if(log != nullptr)
    logger.addOStream(*log);
  return std::make_unique<GameInitializer>(cfg, logger, "testdotssgfstartposes");
}

bool gameInitializerThrows(const string& cfgStr) {
  try {
    createGameInitializer(cfgStr);
  }
  catch(const StringError&) {
    return true;
  }
  return false;
}

}

void Tests::runDotsSgfStartPosesTests() {
  cout << "Running Dots start positions from SGF tests" << endl;

  Rand rand("runDotsSgfStartPosesTests");
  const Rules cross4Rules(Rules::START_POS_CROSS_4, true, false, false, Rules::DEFAULT_DOTS.dotsFreeCapturedDots);
  const Rules crossRules(Rules::START_POS_CROSS, false, false, false, Rules::DEFAULT_DOTS.dotsFreeCapturedDots);
  const vector games = {
    playRandomDotsGame(cross4Rules, 39, 32, 900, rand),
    playRandomDotsGame(crossRules, 39, 32, 600, rand),
  };

  cout << "  Positions of Dots SGFs" << endl;
  for(const DotsGame& game : games)
    checkSamplesOfGame(game);

  cout << "  Selfplay games from Dots .sgf and .sgfs" << endl;
  const gfs::path dir = gfs::temp_directory_path() / ("katago-dots-sgf-startposes-" + Global::uint64ToHexString(rand.nextUInt64()));
  gfs::create_directories(dir);
  {
    ofstream sgfOut;
    FileUtils::open(sgfOut, (dir / "game.sgf").string());
    sgfOut << toSgfLine(games[0].hist) << endl;
    sgfOut.close();

    //One game per line as written by selfplay, the repeated game adds no positions
    ofstream sgfsOut;
    FileUtils::open(sgfsOut, (dir / "games.sgfs").string());
    sgfsOut << toSgfLine(games[1].hist) << endl << toSgfLine(games[0].hist) << endl;
    sgfsOut.close();
  }

  {
    //No startPoses: all games start from SGF positions
    ostringstream log;
    const std::unique_ptr<GameInitializer> gameInit = createGameInitializer(
      "startPosesFromSgfDir = " + dir.generic_string() + "\n"
      "startPosesProb = 1.0\n",
      &log
    );
    //The positions of a game share its sample, the repeated game adds nothing
    const size_t numPositions = games[0].hist.moveHistory.size() + games[1].hist.moveHistory.size();
    testAssert(log.str().find("Kept " + Global::uint64ToString(numPositions) + " start positions in 2 samples") != string::npos);

    vector<bool> gameIsUsed(games.size(), false);
    for(int n = 0; n < 100; n++) {
      Board board;
      Player pla;
      BoardHistory hist;
      ExtraBlackAndKomi extraBlackAndKomi;
      OtherGameProperties otherGameProps;
      gameInit->createGame(board, pla, hist, extraBlackAndKomi, nullptr, PlaySettings(), otherGameProps, nullptr, BoardHistoryModes());
      testAssert(otherGameProps.isSgfPos);

      const size_t gameIdx = hist.initialBoard.start_pos_moves.size() == 16 ? 0 : 1;
      const DotsGame& game = games[gameIdx];
      gameIsUsed[gameIdx] = true;

      //The game goes on under the rules of the SGF, not of the config (dotsCaptureEmptyBases = true)
      Rules expectedRules = game.hist.rules;
      expectedRules.komi = hist.rules.komi;
      testAssert(hist.rules == expectedRules);
      testAssert(board.rules == hist.rules);
      testAssert(hist.initialBoard.rules == hist.rules);
      testAssert(hist.initialTurnNumber == game.hist.initialTurnNumber);
      testAssert(dotsBoardsEqual(hist.initialBoard, game.boards[0], false));

      //The game goes on from its position in the SGF, with the whole history
      testAssert(hist.moveHistory.size() < game.boards.size());
      for(size_t i = 0; i < hist.moveHistory.size(); i++)
        testAssert(movesEqual(hist.moveHistory[i], game.hist.moveHistory[i]));
      testAssert(dotsBoardsEqual(board, game.boards[hist.moveHistory.size()], false));
      testAssert(pla == hist.presumedNextMovePla);

      //The SGF of the game loads back with the same start pos
      const Rules sgfRules = Sgf::parse(WriteSgf::toSgf(hist))->getRulesOrFail();
      testAssert(sgfRules.startPos == game.hist.rules.startPos);
      testAssert(sgfRules.startPosIsRandom == game.hist.rules.startPosIsRandom);
    }
    testAssert(gameIsUsed[0] && gameIsUsed[1]);
  }

  {
    //startPoses for the games that don't start from SGF positions
    const std::unique_ptr<GameInitializer> gameInit = createGameInitializer(
      "startPosesFromSgfDir = " + dir.generic_string() + "\n"
      "startPosesProb = 0.5\n"
      "startPoses = CROSS_2\n"
      "startPosesAreRandom = false\n"
    );

    int numSgfGames = 0;
    for(int n = 0; n < 100; n++) {
      Board board;
      Player pla;
      BoardHistory hist;
      ExtraBlackAndKomi extraBlackAndKomi;
      OtherGameProperties otherGameProps;
      gameInit->createGame(board, pla, hist, extraBlackAndKomi, nullptr, PlaySettings(), otherGameProps, nullptr, BoardHistoryModes());
      if(otherGameProps.isSgfPos) {
        numSgfGames++;
        testAssert(hist.rules.startPos != Rules::START_POS_CROSS_2);
      } else {
        testAssert(hist.rules.startPos == Rules::START_POS_CROSS_2);
        testAssert(hist.rules.dotsCaptureEmptyBases);
        testAssert(hist.moveHistory.empty());
      }
    }
    testAssert(numSgfGames > 0 && numSgfGames < 100);
  }

  cout << "  Configs without startPoses" << endl;
  testAssert(gameInitializerThrows("startPosesFromSgfDir = " + dir.generic_string() + "\nstartPosesProb = 0.5\n"));
  testAssert(gameInitializerThrows(""));
  gfs::remove_all(dir);
  gfs::create_directories(dir);
  testAssert(gameInitializerThrows("startPosesFromSgfDir = " + dir.generic_string() + "\nstartPosesProb = 1.0\n"));
  gfs::remove_all(dir);
}
