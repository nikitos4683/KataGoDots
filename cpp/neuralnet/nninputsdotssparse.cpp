// Sparse Dots position format (model version 18), see docs/DotsSparseNN.md

#include "../neuralnet/nninputs.h"

#include "../core/test.h"

using namespace std;

void DotsSparseInput::clear() {
  xSize = 0;
  ySize = 0;
  std::fill_n(global, NNInputs::NUM_FEATURES_GLOBAL_DOTS_SPARSE, 0.0f);
  points.clear();
  areas.clear();
  vertices.clear();
  vertexPoints.clear();
  candidates.clear();
  groundAllowed = false;
  groundResult = 0.0f;
  pointLocs.clear();
  candidateLocs.clear();
}

void DotsSparseInput::applySymmetry(
  int x, int y, const int xSize, const int ySize, const int symmetry, int& outX, int& outY
) {
  if(SymmetryHelpers::isFlipX(symmetry))
    x = xSize - x - 1;
  if(SymmetryHelpers::isFlipY(symmetry))
    y = ySize - y - 1;
  if(SymmetryHelpers::isTranspose(symmetry))
    std::swap(x, y);
  outX = x;
  outY = y;
}

DotsSparseInput DotsSparseInput::withSymmetry(const int symmetry) const {
  DotsSparseInput result = *this;
  if(symmetry == 0)
    return result;
  auto transform = [&](int8_t& x, int8_t& y) {
    int sx, sy;
    applySymmetry(x, y, xSize, ySize, symmetry, sx, sy);
    x = static_cast<int8_t>(sx);
    y = static_cast<int8_t>(sy);
  };
  for(Point& point : result.points)
    transform(point.x, point.y);
  for(Vertex& vertex : result.vertices)
    transform(vertex.x, vertex.y);
  for(Candidate& candidate : result.candidates)
    transform(candidate.x, candidate.y);
  // Locations are not transformed: they always refer to the original board
  if(SymmetryHelpers::isTranspose(symmetry)) {
    std::swap(result.xSize, result.ySize);
    std::swap(
      result.global[static_cast<int>(NNInputs::DotsSparseGlobalFeature::XSize_6)],
      result.global[static_cast<int>(NNInputs::DotsSparseGlobalFeature::YSize_7)]
    );
  }
  return result;
}

nlohmann::json DotsSparseInput::toJson() const {
  nlohmann::json result;
  result["xSize"] = xSize;
  result["ySize"] = ySize;
  result["global"] = std::vector<float>(global, global + NNInputs::NUM_FEATURES_GLOBAL_DOTS_SPARSE);
  nlohmann::json pointsJson = nlohmann::json::array();
  for(const Point& point : points) {
    pointsJson.push_back({
      {"x", point.x}, {"y", point.y}, {"owner", point.owner}, {"grounded", point.grounded}, {"movesAgo", point.movesAgo}
    });
  }
  result["points"] = pointsJson;
  nlohmann::json areasJson = nlohmann::json::array();
  for(const Area& area : areas) {
    nlohmann::json boundary = nlohmann::json::array();
    for(int i = 0; i < area.boundaryLen; i++) {
      const Vertex& vertex = vertices[area.vertexStart + i];
      boundary.push_back({vertex.x, vertex.y});
    }
    areasJson.push_back({
      {"owner", area.owner}, {"grounded", area.grounded}, {"movesAgo", area.movesAgo},
      {"insideMy", area.insideMy}, {"insideOpp", area.insideOpp}, {"boundaryLen", area.boundaryLen},
      {"boundary", boundary}
    });
  }
  result["areas"] = areasJson;
  nlohmann::json candidatesJson = nlohmann::json::array();
  for(const Candidate& candidate : candidates) {
    candidatesJson.push_back({
      {"x", candidate.x}, {"y", candidate.y}, {"gainMy", candidate.gainMy}, {"gainOpp", candidate.gainOpp},
      {"inMyHouse", candidate.inMyHouse}, {"inOppHouse", candidate.inOppHouse}
    });
  }
  result["candidates"] = candidatesJson;
  result["groundAllowed"] = groundAllowed;
  result["groundResult"] = groundResult;
  return result;
}

// Change of the white minus black score if the player plays at the location.
// Zero for a suicide since suicide is illegal (the only rules the sparse models support).
static int whiteScoreGainOfMove(const Board& board, const Loc loc, const Player pla) {
  // The same trick as in Board::recalculateCapturesAndTerritories: play and undo on the const board
  auto& mutableBoard = const_cast<Board&>(board);
  const int scoreBefore = board.numBlackCaptures - board.numWhiteCaptures;
  const Board::MoveRecord moveRecord = mutableBoard.tryPlayMoveRecordedDots(loc, pla, false);
  if(moveRecord.pla == C_EMPTY)
    return 0;
  const int scoreAfter = board.numBlackCaptures - board.numWhiteCaptures;
  mutableBoard.undoDots(moveRecord);
  return scoreAfter - scoreBefore;
}

void NNInputs::fillDotsSparseInput(
  const Board& board,
  const BoardHistory& hist,
  const Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  DotsSparseInput& result,
  const bool selfplay,
  std::vector<Loc>* outputReasonableMoves
) {
  assert(board.isDots());
  result.clear();

  const Player pla = nextPlayer;
  const Player opp = getOpp(pla);
  const int xSize = board.x_size;
  const int ySize = board.y_size;
  result.xSize = xSize;
  result.ySize = ySize;

  auto setGlobal = [&](const DotsSparseGlobalFeature feature, const float value) {
    result.global[static_cast<int>(feature)] = value;
  };

  const auto capturesAndTerritoriesInfos = board.calculateCapturesAndTerritoriesColorsForDots();
  vector<Loc> reasonableMoves = hist.getReasonableMoves(board, pla, Board::NULL_LOC, false, &capturesAndTerritoriesInfos);
  const bool hasReasonableNonGroundMove = !reasonableMoves.empty() && reasonableMoves.front() != Board::PASS_LOC;

  // History, the same rules as in fillRowV7Dots -----------------------------------------------------------
  int maxTurnsOfHistoryToInclude = std::min(DOTS_SPARSE_MAX_MOVES_AGO, std::max(0, nnInputParams.maxHistory));
  if(!hasReasonableNonGroundMove) {
    // Don't include history for non-resultative games: when there are no reasonable non-ground moves to play
    maxTurnsOfHistoryToInclude = 0;
  }
  const vector<Move>& moveHistory = hist.moveHistory;
  const int amountOfHistoryToTryToUse = std::min(maxTurnsOfHistoryToInclude, hist.numApproxValidTurnsThisPhase);
  const int moveHistoryLen = static_cast<int>(moveHistory.size());
  testAssert(moveHistoryLen >= hist.numApproxValidTurnsThisPhase);

  // movesAgo of the dot placed at the location by the history (0 if none)
  vector<int8_t> movesAgoOfLoc(Board::MAX_ARR_SIZE, 0);
  // Whether the move that many moves ago is marked in the history (index 0 is unused)
  bool movesAgoIsMarked[DOTS_SPARSE_MAX_MOVES_AGO + 1] = {};
  // Number of moves ago examined before the history stopped
  int numExaminedMovesAgo = 0;
  bool groundIsEncountered = false;
  {
    Player currentPla = opp;
    for(int i = 0; i < amountOfHistoryToTryToUse; i++) {
      const int index = moveHistoryLen - i - 1;
      if(index < 0)
        break;
      const Move& prevMove = moveHistory[index];
      // History is only actual for strict moves order (colors should be alternating)
      if(prevMove.pla != currentPla)
        break;
      numExaminedMovesAgo = i + 1;
      const Loc prevLoc = prevMove.loc;
      if(prevLoc == Board::NULL_LOC)
        continue;
      if(prevLoc == Board::PASS_LOC) {
        // Since the grounding is always ending mode, it could be only single one
        testAssert(!groundIsEncountered);
        groundIsEncountered = true;
      } else {
        // Unreasonable moves can exist in a real game (suicides, corners), but not during selfplay training.
        // Ignore them because NN isn't trained on them.
        if(!BoardHistory::isReasonableForDotsNoCapturesAndTerritories(prevLoc, hist.getRecentBoard(i + 1), currentPla)) {
          testAssert(!selfplay && "Unreasonable move during selfplay game!");
          continue;
        }
        movesAgoOfLoc[prevLoc] = static_cast<int8_t>(i + 1);
      }
      movesAgoIsMarked[i + 1] = true;
      currentPla = getOpp(currentPla);
    }
  }
  setGlobal(DotsSparseGlobalFeature::HistoryGround_0, groundIsEncountered ? 1.0f : 0.0f);

  // Points: placed dots outside the interiors of the areas -----------------------------------------------------
  vector<int32_t> pointIdxOfLoc(Board::MAX_ARR_SIZE, -1);
  for(int y = 0; y < ySize; y++) {
    for(int x = 0; x < xSize; x++) {
      const Loc loc = Location::getLoc(x, y, xSize);
      const State state = board.getState(loc);
      const Color placedColor = getPlacedDotColor(state);
      if(placedColor == C_EMPTY || isTerritory(state))
        continue;
      // Outside the areas the dots are never captured
      testAssert(getActiveColor(state) == placedColor);
      DotsSparseInput::Point point{};
      point.x = static_cast<int8_t>(x);
      point.y = static_cast<int8_t>(y);
      point.owner = placedColor == pla ? 0 : 1;
      point.grounded = isGrounded(state) ? 1 : 0;
      point.movesAgo = movesAgoOfLoc[loc];
      pointIdxOfLoc[loc] = static_cast<int32_t>(result.points.size());
      result.points.push_back(point);
      result.pointLocs.push_back(loc);
    }
  }

  // Areas: 4-connected components of the territory locations with the same owner ------------------------------
  // The boundary is the active not covered dots of the owner adjacent to the component.
  // Nested recaptures are already resolved by the engine, so only the outer contours remain.
  {
    vector<int> stamp(Board::MAX_ARR_SIZE, -1);
    vector<int> boundaryStamp(Board::MAX_ARR_SIZE, -1);
    vector<Loc> interior;
    vector<Loc> boundary;
    vector<Loc> stack;
    const int adjOffsets[4] = {-(xSize + 1), -1, 1, xSize + 1};
    for(int y = 0; y < ySize; y++) {
      for(int x = 0; x < xSize; x++) {
        const Loc startLoc = Location::getLoc(x, y, xSize);
        const State startState = board.getState(startLoc);
        if(!isTerritory(startState) || stamp[startLoc] >= 0)
          continue;
        const int areaIndex = static_cast<int>(result.areas.size());
        const Color owner = getActiveColor(startState);
        testAssert(owner == P_BLACK || owner == P_WHITE);

        interior.clear();
        boundary.clear();
        stamp[startLoc] = areaIndex;
        stack.push_back(startLoc);
        while(!stack.empty()) {
          const Loc loc = stack.back();
          stack.pop_back();
          interior.push_back(loc);
          for(const int offset : adjOffsets) {
            const Loc adj = static_cast<Loc>(loc + offset);
            const State adjState = board.getState(adj);
            if(getActiveColor(adjState) != owner)
              continue;
            if(isTerritory(adjState)) {
              if(stamp[adj] < 0) {
                stamp[adj] = areaIndex;
                stack.push_back(adj);
              }
            } else if(boundaryStamp[adj] != areaIndex) {
              boundaryStamp[adj] = areaIndex;
              boundary.push_back(adj);
            }
          }
        }

        int insideMy = 0;
        int insideOpp = 0;
        for(const Loc loc : interior) {
          const Color placedColor = getPlacedDotColor(board.getState(loc));
          if(placedColor == pla)
            insideMy++;
          else if(placedColor == opp)
            insideOpp++;
        }

        // Sort the boundary in the scan order to make it deterministic (the order is not a feature)
        std::sort(boundary.begin(), boundary.end());
        bool grounded = false;
        for(const Loc loc : boundary)
          grounded = grounded || isGrounded(board.getState(loc));

        // The area appeared k moves ago if before that move not all its interior was the territory of the owner
        int movesAgo = 0;
        for(int k = 1; k <= numExaminedMovesAgo; k++) {
          const Board& recentBoard = hist.getRecentBoard(k);
          bool allTerritory = true;
          for(const Loc loc : interior) {
            const State recentState = recentBoard.getState(loc);
            if(!isTerritory(recentState) || getActiveColor(recentState) != owner) {
              allTerritory = false;
              break;
            }
          }
          if(!allTerritory) {
            if(movesAgoIsMarked[k])
              movesAgo = k;
            break;
          }
        }

        DotsSparseInput::Area area{};
        area.owner = owner == pla ? 0 : 1;
        area.grounded = grounded ? 1 : 0;
        area.movesAgo = static_cast<int16_t>(movesAgo);
        area.insideMy = static_cast<int16_t>(insideMy);
        area.insideOpp = static_cast<int16_t>(insideOpp);
        area.boundaryLen = static_cast<int16_t>(boundary.size());
        area.vertexStart = static_cast<int32_t>(result.vertices.size());
        for(const Loc loc : boundary) {
          result.vertices.push_back({
            static_cast<int8_t>(Location::getX(loc, xSize)), static_cast<int8_t>(Location::getY(loc, xSize))
          });
          // The boundary dots are active dots of the owner outside the interiors, so they are points
          const int32_t pointIdx = pointIdxOfLoc[loc];
          testAssert(pointIdx >= 0);
          result.vertexPoints.push_back(pointIdx);
          result.points[pointIdx].boundary = 1;
        }
        result.areas.push_back(area);
      }
    }
  }

  // Candidates ----------------------------------------------------------------------------------------------
  for(const Loc loc : reasonableMoves) {
    if(loc == Board::PASS_LOC) {
      result.groundAllowed = true;
      continue;
    }
    const State state = board.getState(loc);
    DotsSparseInput::Candidate candidate{};
    candidate.x = static_cast<int8_t>(Location::getX(loc, xSize));
    candidate.y = static_cast<int8_t>(Location::getY(loc, xSize));
    const Color emptyTerritoryColor = getEmptyTerritoryColor(state);
    candidate.inMyHouse = emptyTerritoryColor == pla ? 1 : 0;
    candidate.inOppHouse = emptyTerritoryColor == opp ? 1 : 0;
    // Only the moves that surround something change the score. A move tries to surround under the same condition
    // as here (at least 2 unconnected neighbor dots). CapturesAndTerritoriesInfos can't be used: it skips the captures
    // by moves inside the own potential territory of the player.
    const Color potentialCapturingColor = board.getColorsOfPotentialCapturing(loc);
    if((potentialCapturingColor & pla) != 0) {
      const int whiteGain = whiteScoreGainOfMove(board, loc, pla);
      candidate.gainMy = static_cast<int16_t>(pla == P_WHITE ? whiteGain : -whiteGain);
    }
    if((potentialCapturingColor & opp) != 0) {
      const int whiteGain = whiteScoreGainOfMove(board, loc, opp);
      candidate.gainOpp = static_cast<int16_t>(opp == P_WHITE ? whiteGain : -whiteGain);
    }
    result.candidates.push_back(candidate);
    result.candidateLocs.push_back(loc);
  }

  // Scores and komi -----------------------------------------------------------------------------------------
  float selfKomi = hist.currentSelfKomi(pla, nnInputParams.drawEquivalentWinsForWhite);
  const auto bArea = static_cast<float>(xSize * ySize);
  // Bound komi just in case
  if(selfKomi > bArea + NNPos::KOMI_CLIP_RADIUS)
    selfKomi = bArea + NNPos::KOMI_CLIP_RADIUS;
  if(selfKomi < -bArea - NNPos::KOMI_CLIP_RADIUS)
    selfKomi = -bArea - NNPos::KOMI_CLIP_RADIUS;

  // numBlackCaptures is the number of captured black dots, i.e. the score of white
  const int whiteScore = board.numBlackCaptures;
  const int blackScore = board.numWhiteCaptures;
  const int scoreMy = pla == P_WHITE ? whiteScore : blackScore;
  const int scoreOpp = pla == P_WHITE ? blackScore : whiteScore;

  setGlobal(DotsSparseGlobalFeature::Komi_1, selfKomi / NNPos::KOMI_CLIP_RADIUS);
  if(!hasReasonableNonGroundMove || hist.isGroundReasonable(board, pla)) {
    setGlobal(DotsSparseGlobalFeature::GroundReasonable_2, 1.0f);
  }
  setGlobal(DotsSparseGlobalFeature::ScoreMy_3, static_cast<float>(scoreMy) / NNPos::KOMI_CLIP_RADIUS);
  setGlobal(DotsSparseGlobalFeature::ScoreOpp_4, static_cast<float>(scoreOpp) / NNPos::KOMI_CLIP_RADIUS);
  setGlobal(
    DotsSparseGlobalFeature::ScoreDiffPlusKomi_5,
    (static_cast<float>(scoreMy - scoreOpp) + selfKomi) / NNPos::KOMI_CLIP_RADIUS
  );
  setGlobal(DotsSparseGlobalFeature::XSize_6, static_cast<float>(xSize) / 40.0f);
  setGlobal(DotsSparseGlobalFeature::YSize_7, static_cast<float>(ySize) / 40.0f);

  if(result.groundAllowed) {
    // whiteScoreIfBlackGrounds is the final white minus black score if black grounds now and vice versa
    const int whiteScoreAfterGrounding =
      pla == P_BLACK ? board.whiteScoreIfBlackGrounds : -board.blackScoreIfWhiteGrounds;
    const int myScoreAfterGrounding = pla == P_WHITE ? whiteScoreAfterGrounding : -whiteScoreAfterGrounding;
    result.groundResult = static_cast<float>(myScoreAfterGrounding) + selfKomi;
  }

  if(outputReasonableMoves != nullptr)
    *outputReasonableMoves = std::move(reasonableMoves);
}
