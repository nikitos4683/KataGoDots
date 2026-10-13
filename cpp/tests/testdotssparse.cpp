// Tests of the sparse Dots position format (model version 18), see docs/DotsSparseNN.md

#include "../tests/tests.h"
#include "../tests/testdotsutils.h"

#include "../neuralnet/nninputs.h"

using namespace std;
using namespace TestCommon;

namespace {

DotsSparseInput fillSparse(const Board& board, const BoardHistory& hist, const Player pla, vector<Loc>* reasonableMoves = nullptr) {
  DotsSparseInput result;
  MiscNNInputParams params;
  NNInputs::fillDotsSparseInput(board, hist, pla, params, result, true, reasonableMoves);
  return result;
}

// Order independent representation for comparisons
struct NormalizedSparseInput {
  int xSize;
  int ySize;
  vector<float> global;
  vector<tuple<int, int, int, int, int>> points;
  vector<tuple<int, int, int, int, int, int, vector<pair<int, int>>>> areas;
  vector<tuple<int, int, int, int, int, int>> candidates;
  bool groundAllowed;
  float groundResult;

  explicit NormalizedSparseInput(const DotsSparseInput& input) {
    xSize = input.xSize;
    ySize = input.ySize;
    global.assign(input.global, input.global + NNInputs::NUM_FEATURES_GLOBAL_DOTS_SPARSE);
    for(const auto& p : input.points)
      points.emplace_back(p.x, p.y, p.owner, p.grounded, p.movesAgo);
    for(const auto& a : input.areas) {
      vector<pair<int, int>> boundary;
      for(int i = 0; i < a.boundaryLen; i++)
        boundary.emplace_back(input.vertices[a.vertexStart + i].x, input.vertices[a.vertexStart + i].y);
      std::sort(boundary.begin(), boundary.end());
      areas.emplace_back(a.owner, a.grounded, a.movesAgo, a.insideMy, a.insideOpp, a.boundaryLen, boundary);
    }
    for(const auto& c : input.candidates)
      candidates.emplace_back(c.x, c.y, c.gainMy, c.gainOpp, c.inMyHouse, c.inOppHouse);
    std::sort(points.begin(), points.end());
    std::sort(areas.begin(), areas.end());
    std::sort(candidates.begin(), candidates.end());
    groundAllowed = input.groundAllowed;
    groundResult = input.groundResult;
  }

  bool operator==(const NormalizedSparseInput& other) const {
    return xSize == other.xSize && ySize == other.ySize && global == other.global && points == other.points &&
           areas == other.areas && candidates == other.candidates && groundAllowed == other.groundAllowed &&
           groundResult == other.groundResult;
  }
};

set<Loc> getAreaBoundaryLocs(const DotsSparseInput& input, const DotsSparseInput::Area& area) {
  set<Loc> result;
  for(int i = 0; i < area.boundaryLen; i++) {
    const auto& v = input.vertices[area.vertexStart + i];
    result.insert(Location::getLoc(v.x, v.y, input.xSize));
  }
  return result;
}

// Interior locations of the area containing the location (4-connected territory of the same owner)
set<Loc> getAreaInterior(const Board& board, const Loc startLoc) {
  set<Loc> result;
  const Color owner = getActiveColor(board.getState(startLoc));
  vector<Loc> stack = {startLoc};
  result.insert(startLoc);
  while(!stack.empty()) {
    const Loc loc = stack.back();
    stack.pop_back();
    for(const int offset : {-(board.x_size + 1), -1, 1, board.x_size + 1}) {
      const Loc adj = static_cast<Loc>(loc + offset);
      const State state = board.getState(adj);
      if(isTerritory(state) && getActiveColor(state) == owner && result.insert(adj).second)
        stack.push_back(adj);
    }
  }
  return result;
}

int myMinusOppScore(const Board& board, const Player pla) {
  const int whiteMinusBlack = board.numBlackCaptures - board.numWhiteCaptures;
  return pla == P_WHITE ? whiteMinusBlack : -whiteMinusBlack;
}

// Checks the invariants of the sparse input that don't depend on the history of the game
void checkSparseInvariants(const Board& board, const BoardHistory& hist, const Player pla, const DotsSparseInput& input, const vector<Loc>& reasonableMoves) {
  const Player opp = getOpp(pla);

  // Candidates are exactly the reasonable moves
  {
    vector<Loc> expected;
    bool groundExpected = false;
    for(const Loc loc : reasonableMoves) {
      if(loc == Board::PASS_LOC)
        groundExpected = true;
      else
        expected.push_back(loc);
    }
    testAssert(expected == input.candidateLocs);
    testAssert(groundExpected == input.groundAllowed);
    testAssert(input.candidates.size() == input.candidateLocs.size());
    for(size_t i = 0; i < input.candidates.size(); i++) {
      const auto& c = input.candidates[i];
      testAssert(Location::getLoc(c.x, c.y, board.x_size) == input.candidateLocs[i]);
    }
  }

  // The area features of a point aggregate the areas whose boundary has the dot: their number, the sums of
  // the dots inside, the freshest history mark. A boundary dot belongs to the owner of the area.
  {
    vector<int> areaCount(input.points.size(), 0), insideMy(input.points.size(), 0), insideOpp(input.points.size(), 0);
    vector<int> areaMovesAgo(input.points.size(), 0);
    for(const auto& area : input.areas) {
      for(int i = 0; i < area.boundaryLen; i++) {
        const auto& vertex = input.vertices[area.vertexStart + i];
        int pointIdx = -1;
        for(size_t j = 0; j < input.points.size(); j++)
          if(input.points[j].x == vertex.x && input.points[j].y == vertex.y)
            pointIdx = (int)j;
        testAssert(pointIdx >= 0);
        testAssert(input.points[pointIdx].owner == area.owner);
        areaCount[pointIdx]++;
        insideMy[pointIdx] += area.insideMy;
        insideOpp[pointIdx] += area.insideOpp;
        if(area.movesAgo != 0 && (areaMovesAgo[pointIdx] == 0 || area.movesAgo < areaMovesAgo[pointIdx]))
          areaMovesAgo[pointIdx] = area.movesAgo;
      }
    }
    for(size_t j = 0; j < input.points.size(); j++) {
      const auto& point = input.points[j];
      testAssert(point.areaCount == areaCount[j]);
      testAssert(point.insideMy == insideMy[j]);
      testAssert(point.insideOpp == insideOpp[j]);
      testAssert(point.areaMovesAgo == areaMovesAgo[j]);
    }
  }

  // Sum of the opponent dots inside my areas = my captures and vice versa
  {
    int insideOppOfMyAreas = 0;
    int insideMyOfOppAreas = 0;
    int numInteriorDots = 0;
    for(const auto& area : input.areas) {
      if(area.owner == 0)
        insideOppOfMyAreas += area.insideOpp;
      else
        insideMyOfOppAreas += area.insideMy;
      numInteriorDots += area.insideMy + area.insideOpp;
    }
    const int myCaptures = pla == P_WHITE ? board.numBlackCaptures : board.numWhiteCaptures;
    const int oppCaptures = pla == P_WHITE ? board.numWhiteCaptures : board.numBlackCaptures;
    testAssert(insideOppOfMyAreas == myCaptures);
    testAssert(insideMyOfOppAreas == oppCaptures);

    // Every placed dot is either a point token or inside exactly one area
    int numPlacedDots = 0;
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        if(getPlacedDotColor(board.getState(Location::getLoc(x, y, board.x_size))) != C_EMPTY)
          numPlacedDots++;
      }
    }
    testAssert(numPlacedDots == static_cast<int>(input.points.size()) + numInteriorDots);
  }

  // The grounding bit is the same on the whole boundary of an area, and the boundary is made of the owner's points
  {
    set<Loc> pointLocs(input.pointLocs.begin(), input.pointLocs.end());
    for(const auto& area : input.areas) {
      testAssert(area.boundaryLen >= 4);
      const set<Loc> boundary = getAreaBoundaryLocs(input, area);
      testAssert(static_cast<int>(boundary.size()) == area.boundaryLen);
      const Color owner = area.owner == 0 ? pla : opp;
      for(const Loc loc : boundary) {
        const State state = board.getState(loc);
        testAssert(pointLocs.count(loc) == 1);
        testAssert(getActiveColor(state) == owner && getPlacedDotColor(state) == owner && !isTerritory(state));
        testAssert(isGrounded(state) == (area.grounded != 0));
      }
    }
  }

  // Points
  for(size_t i = 0; i < input.points.size(); i++) {
    const auto& point = input.points[i];
    const Loc loc = input.pointLocs[i];
    testAssert(Location::getLoc(point.x, point.y, board.x_size) == loc);
    const State state = board.getState(loc);
    testAssert(getPlacedDotColor(state) == (point.owner == 0 ? pla : opp));
    testAssert(isGrounded(state) == (point.grounded != 0));
  }

  // Grounding result is the real result of grounding now
  if(input.groundAllowed) {
    Board copy(board);
    BoardHistory histCopy(hist);
    histCopy.makeBoardMoveAssumeLegal(copy, Board::PASS_LOC, pla, nullptr);
    testAssert(histCopy.isGameFinished);
    const float myFinalScore = pla == P_WHITE ? histCopy.finalWhiteMinusBlackScore : -histCopy.finalWhiteMinusBlackScore;
    const float drawAdjustment = hist.currentSelfKomi(pla, 0.5) - (pla == P_WHITE ? hist.getCompleteWhiteBonus() : -hist.getCompleteWhiteBonus());
    testAssert(std::abs(input.groundResult - (myFinalScore + drawAdjustment)) < 1e-5);
  }

  // Global features
  {
    const int myCaptures = pla == P_WHITE ? board.numBlackCaptures : board.numWhiteCaptures;
    const int oppCaptures = pla == P_WHITE ? board.numWhiteCaptures : board.numBlackCaptures;
    testAssert(input.global[3] == static_cast<float>(myCaptures) / 20.0f);
    testAssert(input.global[4] == static_cast<float>(oppCaptures) / 20.0f);
    testAssert(std::abs(input.global[5] - (static_cast<float>(myCaptures - oppCaptures) / 20.0f + input.global[1])) < 1e-5);
    testAssert(input.global[6] == static_cast<float>(board.x_size) / 40.0f);
    testAssert(input.global[7] == static_cast<float>(board.y_size) / 40.0f);
  }
}

}  // namespace

static void runDotsSparseManualTests() {
  cout << "  Manual positions" << endl;

  // Simple capture and the history of points and areas
  {
    Board board = parseDotsFieldDefault(R"(
. . . . . .
. . x . . .
. x o x . .
. . . . o .
. . . . . .
)");
    BoardHistory hist(board, P_BLACK, board.rules, 0, BoardHistoryModes());
    Board recordBoard(board);
    const Loc captureLoc = Location::getLoc(2, 3, board.x_size);
    const Board::MoveRecord record = recordBoard.playMoveRecordedDots(captureLoc, P_BLACK);
    hist.makeBoardMoveAssumeLegal(board, captureLoc, P_BLACK, nullptr);

    vector<Loc> reasonableMoves;
    const DotsSparseInput input = fillSparse(board, hist, P_WHITE, &reasonableMoves);
    checkSparseInvariants(board, hist, P_WHITE, input, reasonableMoves);

    // 4 black points of the boundary and 1 white point outside, the captured white dot is not a token
    testAssert(input.points.size() == 5);
    testAssert(input.areas.size() == 1);
    const auto& area = input.areas[0];
    testAssert(area.owner == 1); // black is the opponent of white to move
    testAssert(area.insideMy == 1);
    testAssert(area.insideOpp == 0);
    testAssert(area.boundaryLen == 4);
    testAssert(area.movesAgo == 1);
    testAssert(record.bases.size() == 1);
    const set<Loc> expectedBoundary(record.bases[0].surrounding_locs.begin(), record.bases[0].surrounding_locs.end());
    testAssert(getAreaBoundaryLocs(input, area) == expectedBoundary);
    for(size_t i = 0; i < input.points.size(); i++) {
      testAssert(input.points[i].movesAgo == (input.pointLocs[i] == captureLoc ? 1 : 0));
    }
    testAssert(input.global[3] == 0.0f);
    testAssert(input.global[4] == 1.0f / 20.0f);

    // Without the history the marks are absent
    DotsSparseInput noHistory;
    MiscNNInputParams params;
    params.maxHistory = 0;
    NNInputs::fillDotsSparseInput(board, hist, P_WHITE, params, noHistory, true);
    testAssert(noHistory.areas[0].movesAgo == 0);
    for(const auto& point : noHistory.points)
      testAssert(point.movesAgo == 0);

    // One more move: the marks shift
    const Loc whiteLoc = Location::getLoc(5, 1, board.x_size);
    hist.makeBoardMoveAssumeLegal(board, whiteLoc, P_WHITE, nullptr);
    const DotsSparseInput input2 = fillSparse(board, hist, P_BLACK);
    testAssert(input2.areas.size() == 1 && input2.areas[0].movesAgo == 2 && input2.areas[0].owner == 0);
    for(size_t i = 0; i < input2.points.size(); i++) {
      const Loc loc = input2.pointLocs[i];
      testAssert(input2.points[i].movesAgo == (loc == whiteLoc ? 1 : loc == captureLoc ? 2 : 0));
    }
  }

  // Nested recapture: only the outer contour remains, every dot inside is counted once by its color
  {
    Board board = parseDotsFieldDefault(R"(
. x x x x x x x .
x . . o o o . . x
x . o . x . o . x
x . o x o x o . x
x . o . . . o . x
x . . o . o . . x
. x x x . x x x .
)");
    BoardHistory hist(board, P_BLACK, board.rules, 0, BoardHistoryModes());
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(4, 4, board.x_size), P_BLACK, nullptr);
    {
      const DotsSparseInput input = fillSparse(board, hist, P_WHITE);
      testAssert(input.areas.size() == 1);
      testAssert(input.areas[0].owner == 1 && input.areas[0].insideMy == 1 && input.areas[0].insideOpp == 0);
    }
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(4, 5, board.x_size), P_WHITE, nullptr);
    {
      // The white area contains the black area: 4 black dots (the old boundary) and 1 white dot inside
      const DotsSparseInput input = fillSparse(board, hist, P_BLACK);
      testAssert(input.areas.size() == 1);
      const auto& area = input.areas[0];
      testAssert(area.owner == 1 && area.insideMy == 4 && area.insideOpp == 1 && area.movesAgo == 1);
      testAssert(board.numBlackCaptures == 4 && board.numWhiteCaptures == 0);
    }
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(4, 6, board.x_size), P_BLACK, nullptr);
    {
      vector<Loc> reasonableMoves;
      const DotsSparseInput input = fillSparse(board, hist, P_WHITE, &reasonableMoves);
      checkSparseInvariants(board, hist, P_WHITE, input, reasonableMoves);
      testAssert(input.areas.size() == 1);
      const auto& area = input.areas[0];
      testAssert(area.owner == 1);
      testAssert(area.insideMy == 13); // all white dots
      testAssert(area.insideOpp == 4); // black dots inside: the boundary of the first black area
      // White has no reasonable moves except grounding (only corners are left), so the history isn't provided
      testAssert(input.candidates.empty() && input.groundAllowed);
      testAssert(area.movesAgo == 0);
      testAssert(board.numWhiteCaptures == 13 && board.numBlackCaptures == 0);
    }
  }

  // Minimal recapture of a single area: the change for the area owner is -(nX + nO + |B|)
  {
    Rules rules = Rules::DEFAULT_DOTS;
    rules.startPos = Rules::START_POS_EMPTY;
    Board board(9, 9, rules);
    BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
    // Black surrounds a white dot, then white builds a diamond tightly around the black area
    const vector<XYMove> moves = {
      XYMove(3, 2, P_BLACK), XYMove(3, 3, P_WHITE), XYMove(2, 3, P_BLACK), XYMove(7, 1, P_WHITE),
      XYMove(4, 3, P_BLACK), XYMove(3, 1, P_WHITE), XYMove(3, 4, P_BLACK), XYMove(2, 2, P_WHITE),
      XYMove(1, 7, P_BLACK), XYMove(4, 2, P_WHITE), XYMove(2, 7, P_BLACK), XYMove(1, 3, P_WHITE),
      XYMove(3, 7, P_BLACK), XYMove(5, 3, P_WHITE), XYMove(4, 7, P_BLACK), XYMove(2, 4, P_WHITE),
      XYMove(5, 7, P_BLACK), XYMove(4, 4, P_WHITE), XYMove(6, 7, P_BLACK),
    };
    for(const XYMove& move : moves)
      hist.makeBoardMoveAssumeLegal(board, Location::getLoc(move.x, move.y, board.x_size), move.player, nullptr);
    testAssert(board.numWhiteCaptures == 1 && board.numBlackCaptures == 0);
    const DotsSparseInput before = fillSparse(board, hist, P_WHITE);
    testAssert(before.areas.size() == 1);
    const auto& area = before.areas[0];
    testAssert(area.owner == 1 && area.insideMy == 1 && area.insideOpp == 0 && area.boundaryLen == 4);
    const int blackScoreBefore = myMinusOppScore(board, P_BLACK);
    hist.makeBoardMoveAssumeLegal(board, Location::getLoc(3, 5, board.x_size), P_WHITE, nullptr);
    const int blackChange = myMinusOppScore(board, P_BLACK) - blackScoreBefore;
    // nX = white dots inside the black area, nO = black dots inside
    testAssert(blackChange == -(area.insideMy + area.insideOpp + area.boundaryLen));
    vector<Loc> reasonableMoves;
    const DotsSparseInput after = fillSparse(board, hist, P_BLACK, &reasonableMoves);
    checkSparseInvariants(board, hist, P_BLACK, after, reasonableMoves);
    testAssert(after.areas.size() == 1);
    testAssert(after.areas[0].owner == 1 && after.areas[0].insideMy == 4 && after.areas[0].insideOpp == 1);
    testAssert(after.areas[0].boundaryLen == 8 && after.areas[0].movesAgo == 1);
  }

  // Several areas by a single move (5.9)
  {
    Board board = parseDotsFieldDefault(R"(
. . . . . . .
. . x . x . .
. x o . o x .
. . x o x . .
. . . x . . .
. . . . . . .
)");
    BoardHistory hist(board, P_BLACK, board.rules, 0, BoardHistoryModes());
    const Loc loc = Location::getLoc(3, 2, board.x_size);
    Board recordBoard(board);
    const Board::MoveRecord record = recordBoard.playMoveRecordedDots(loc, P_BLACK);
    hist.makeBoardMoveAssumeLegal(board, loc, P_BLACK, nullptr);
    vector<Loc> reasonableMoves;
    const DotsSparseInput input = fillSparse(board, hist, P_WHITE, &reasonableMoves);
    checkSparseInvariants(board, hist, P_WHITE, input, reasonableMoves);
    testAssert(input.areas.size() == 3);
    testAssert(record.bases.size() == 3);
    set<set<Loc>> expectedBoundaries;
    for(const auto& base : record.bases)
      expectedBoundaries.insert(set<Loc>(base.surrounding_locs.begin(), base.surrounding_locs.end()));
    set<set<Loc>> boundaries;
    for(const auto& area : input.areas) {
      testAssert(area.movesAgo == 1 && area.owner == 1 && area.insideMy == 1 && area.insideOpp == 0);
      boundaries.insert(getAreaBoundaryLocs(input, area));
    }
    testAssert(boundaries == expectedBoundaries);
    // The move point is shared by the 3 boundaries
    for(const auto& boundary : boundaries)
      testAssert(boundary.count(loc) == 1);
  }

  // Houses: own house is a candidate, the opponent's one is not (suicide is illegal)
  {
    Board board = parseDotsFieldDefault(R"(
. . . . . . .
. . . x . . .
. . x . x . .
. . . x . . .
. . . . . o .
. . . . . . .
)");
    testAssert(!board.rules.multiStoneSuicideLegal);
    BoardHistory hist(board, P_BLACK, board.rules, 0, BoardHistoryModes());
    const Loc houseLoc = Location::getLoc(3, 2, board.x_size);
    testAssert(getEmptyTerritoryColor(board.getState(houseLoc)) == P_BLACK);
    {
      vector<Loc> reasonableMoves;
      const DotsSparseInput input = fillSparse(board, hist, P_BLACK, &reasonableMoves);
      checkSparseInvariants(board, hist, P_BLACK, input, reasonableMoves);
      bool found = false;
      for(size_t i = 0; i < input.candidates.size(); i++) {
        if(input.candidateLocs[i] == houseLoc) {
          found = true;
          testAssert(input.candidates[i].inMyHouse == 1 && input.candidates[i].inOppHouse == 0);
        } else {
          testAssert(input.candidates[i].inMyHouse == 0 && input.candidates[i].inOppHouse == 0);
        }
      }
      testAssert(found);
    }
    {
      hist.setPresumedNextMovePla(P_WHITE);
      vector<Loc> reasonableMoves;
      const DotsSparseInput input = fillSparse(board, hist, P_WHITE, &reasonableMoves);
      checkSparseInvariants(board, hist, P_WHITE, input, reasonableMoves);
      for(const Loc loc : input.candidateLocs)
        testAssert(loc != houseLoc);
    }
  }

  // Gains: a capture threat of each player
  {
    Board board = parseDotsFieldDefault(R"(
. . . . . . . .
. . x . . o . .
. x o x o x o .
. . . . . . . .
. . . . . . . .
)");
    BoardHistory hist(board, P_BLACK, board.rules, 0, BoardHistoryModes());
    vector<Loc> reasonableMoves;
    const DotsSparseInput input = fillSparse(board, hist, P_BLACK, &reasonableMoves);
    checkSparseInvariants(board, hist, P_BLACK, input, reasonableMoves);
    for(size_t i = 0; i < input.candidates.size(); i++) {
      const auto& c = input.candidates[i];
      const bool blackCaptures = c.x == 2 && c.y == 3;
      const bool whiteCaptures = c.x == 5 && c.y == 3;
      testAssert(c.gainMy == (blackCaptures ? 1 : 0));
      testAssert(c.gainOpp == (whiteCaptures ? 1 : 0));
    }
  }

  // Grounding
  {
    Board board = parseDotsFieldDefault(R"(
x . . . .
o x . . .
. . . o .
. . . . .
)");
    BoardHistory hist(board, P_WHITE, board.rules, 0, BoardHistoryModes());
    vector<Loc> reasonableMoves;
    const DotsSparseInput input = fillSparse(board, hist, P_WHITE, &reasonableMoves);
    checkSparseInvariants(board, hist, P_WHITE, input, reasonableMoves);
  }

  // Only the move making the largest surrounding is a candidate: the move at (2,2) surrounds each white dot
  // separately, the move at (2,4) surrounds all of them, and (2,2), (2,3) lie inside that surrounding
  {
    Board board = parseDotsFieldDefault(R"(
. . x . .
. x o x .
x o . o x
. x . x .
. . . . .
)");
    BoardHistory hist(board, P_BLACK, board.rules, 0, BoardHistoryModes());
    vector<Loc> reasonableMoves;
    const DotsSparseInput input = fillSparse(board, hist, P_BLACK, &reasonableMoves);
    checkSparseInvariants(board, hist, P_BLACK, input, reasonableMoves);
    auto candidateIndex = [&](int x, int y) {
      const Loc loc = Location::getLoc(x, y, board.x_size);
      for(size_t i = 0; i < input.candidateLocs.size(); i++)
        if(input.candidateLocs[i] == loc)
          return (int)i;
      return -1;
    };
    testAssert(candidateIndex(2, 2) < 0);
    testAssert(candidateIndex(2, 3) < 0);
    const int largest = candidateIndex(2, 4);
    testAssert(largest >= 0);
    testAssert(input.candidates[largest].gainMy == 3);
  }
}

static void runDotsSparseRandomGames(const int xSize, const int ySize, const int numGames, const int symmetry, const int startPos, const uint64_t seed) {
  cout << "  Random games " << xSize << "x" << ySize << ", symmetry " << symmetry << ", start pos " << Rules::writeStartPosRule(startPos) << endl;
  Rand rand(seed);

  Rules rules = Rules::DEFAULT_DOTS;
  rules.startPos = startPos;
  rules.startPosIsRandom = false;

  int numPositions = 0;
  int numMinimalRecaptures = 0;
  int numCheckedBaseBoundaries = 0;
  int numGains = 0;
  int numAreas = 0;
  int numGroundings = 0;

  for(int game = 0; game < numGames; game++) {
    Board board(xSize, ySize, rules);
    BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
    Board symBoard = SymmetryHelpers::getSymBoard(board, symmetry);
    BoardHistory symHist(symBoard, P_BLACK, rules, 0, BoardHistoryModes());
    Player pla = P_BLACK;
    const int maxMoves = xSize * ySize;

    for(int moveIdx = 0; moveIdx < maxMoves && !hist.isGameFinished; moveIdx++) {
      vector<Loc> reasonableMoves;
      const DotsSparseInput input = fillSparse(board, hist, pla, &reasonableMoves);
      checkSparseInvariants(board, hist, pla, input, reasonableMoves);
      numPositions++;
      numAreas += static_cast<int>(input.areas.size());

      // Equivariance: extraction from the transformed game = transformed extraction
      {
        const DotsSparseInput symInput = fillSparse(symBoard, symHist, pla);
        testAssert(NormalizedSparseInput(symInput) == NormalizedSparseInput(input.withSymmetry(symmetry)));
      }

      if(reasonableMoves.empty())
        break;

      // Choose a move: grounding when allowed with some probability late in the game, otherwise a random candidate
      Loc loc;
      const bool canGround = input.groundAllowed;
      if(input.candidates.empty() || (canGround && rand.nextBool(0.02 + 0.5 * moveIdx / maxMoves))) {
        testAssert(canGround);
        loc = Board::PASS_LOC;
      } else {
        // Prefer captures to make the games more interesting
        vector<int> captureIdxs;
        for(int i = 0; i < static_cast<int>(input.candidates.size()); i++) {
          if(input.candidates[i].gainMy != 0)
            captureIdxs.push_back(i);
        }
        int idx;
        if(!captureIdxs.empty() && rand.nextBool(0.5))
          idx = captureIdxs[rand.nextUInt(static_cast<uint32_t>(captureIdxs.size()))];
        else
          idx = static_cast<int>(rand.nextUInt(static_cast<uint32_t>(input.candidates.size())));
        loc = input.candidateLocs[idx];

        // The gain of the move equals the real change of the score
        Board recordBoard(board);
        const int scoreBefore = myMinusOppScore(recordBoard, pla);
        const Board::MoveRecord record = recordBoard.playMoveRecordedDots(loc, pla);
        const int realGain = myMinusOppScore(recordBoard, pla) - scoreBefore;
        testAssert(realGain == input.candidates[idx].gainMy);
        if(realGain != 0)
          numGains++;

        // Opponent's gain at the location equals the real change if the opponent played there
        {
          Board oppBoard(board);
          if(oppBoard.isLegal(loc, getOpp(pla), true, false) && getEmptyTerritoryColor(board.getState(loc)) != pla) {
            const int oppBefore = myMinusOppScore(oppBoard, getOpp(pla));
            oppBoard.playMoveAssumeLegal(loc, getOpp(pla));
            const int realOppGain = myMinusOppScore(oppBoard, getOpp(pla)) - oppBefore;
            if(realOppGain != input.candidates[idx].gainOpp) {
              cout << "Opp gain mismatch at " << Location::toString(loc, board) << " pla " << PlayerIO::playerToString(pla, true)
                   << " real " << realOppGain << " computed " << input.candidates[idx].gainOpp << endl;
              Board::printBoard(cout, board, loc, &hist.moveHistory);
              Board::printBoard(cout, oppBoard, loc, nullptr);
            }
            testAssert(realOppGain == input.candidates[idx].gainOpp);
          }
        }

        // Boundaries of the created bases match the engine's surrounding locations when the base remains an area
        int numRealBases = 0;
        for(const auto& base : record.bases) {
          if(base.type == Board::Base::Type::EMPTY)
            continue;
          numRealBases++;
          const set<Loc> territory = [&] {
            set<Loc> s;
            for(const auto& r : base.rollback_locs_states_captures)
              s.insert(r.getLoc());
            return s;
          }();
          if(!territory.empty() && getAreaInterior(recordBoard, *territory.begin()) == territory) {
            const set<Loc> expectedBoundary(base.surrounding_locs.begin(), base.surrounding_locs.end());
            bool found = false;
            BoardHistory afterHist(hist);
            Board tmpBoard(board);
            afterHist.makeBoardMoveAssumeLegal(tmpBoard, loc, pla, nullptr);
            const DotsSparseInput afterInput = fillSparse(tmpBoard, afterHist, getOpp(pla));
            for(const auto& area : afterInput.areas) {
              if(getAreaBoundaryLocs(afterInput, area) == expectedBoundary) {
                found = true;
                // Without reasonable moves except grounding the history is not provided (docs/DotsSparseNN.md 3)
                testAssert(area.movesAgo == (afterInput.candidates.empty() ? 0 : 1));
              }
            }
            testAssert(found);
            numCheckedBaseBoundaries++;
          }
        }

        // A recapture of exactly one area by the minimal contour: the change for the area owner is -(nX + nO + |B|)
        if(numRealBases == 1) {
          const Board::Base* base = nullptr;
          for(const auto& b : record.bases)
            if(b.type != Board::Base::Type::EMPTY)
              base = &b;
          set<Loc> territory;
          for(const auto& r : base->rollback_locs_states_captures)
            territory.insert(r.getLoc());
          // Old territory cells inside the new base
          set<Loc> oldTerritory;
          for(const Loc l : territory) {
            if(isTerritory(board.getState(l)))
              oldTerritory.insert(l);
          }
          if(!oldTerritory.empty()) {
            const set<Loc> interior = getAreaInterior(board, *oldTerritory.begin());
            for(const auto& area : input.areas) {
              const set<Loc> boundary = getAreaBoundaryLocs(input, area);
              // The opponent's area surrounded exactly
              if(area.owner == 1 && interior == oldTerritory) {
                set<Loc> expected(interior);
                expected.insert(boundary.begin(), boundary.end());
                if(expected == territory) {
                  // nX: mover's dots inside, nO: area owner's dots inside
                  const int nX = area.insideMy;
                  const int nO = area.insideOpp;
                  const int ownerChange = -realGain;
                  testAssert(ownerChange == -(nX + nO + area.boundaryLen));
                  numMinimalRecaptures++;
                }
              }
            }
          }
        }
      }

      if(loc == Board::PASS_LOC)
        numGroundings++;
      hist.makeBoardMoveAssumeLegal(board, loc, pla, nullptr);
      symHist.makeBoardMoveAssumeLegal(symBoard, SymmetryHelpers::getSymLoc(loc, xSize, ySize, symmetry), pla, nullptr);
      pla = getOpp(pla);
    }
  }

  cout << "    Positions: " << numPositions << ", areas: " << numAreas << ", gains: " << numGains
       << ", checked base boundaries: " << numCheckedBaseBoundaries << ", minimal recaptures: " << numMinimalRecaptures
       << ", groundings: " << numGroundings << endl;
}

void Tests::runDotsSparseInputTests() {
  cout << "Running Dots sparse NN input tests" << endl;
  runDotsSparseManualTests();
  runDotsSparseRandomGames(12, 12, 40, SymmetryHelpers::SYMMETRY_TRANSPOSE_FLIP_X, Rules::START_POS_CROSS, 1);
  runDotsSparseRandomGames(16, 11, 40, SymmetryHelpers::SYMMETRY_FLIP_Y_X, Rules::START_POS_EMPTY, 2);
  runDotsSparseRandomGames(39, 32, 6, SymmetryHelpers::SYMMETRY_FLIP_X, Rules::START_POS_CROSS_4, 3);
}
