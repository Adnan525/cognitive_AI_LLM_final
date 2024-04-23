import chess
import chess.engine
import chess.pgn

from send_move_api import url, send_moves

def send_move(board, engine):
    result = engine.play(board, chess.engine.Limit(time=0.1))
    uci = result.move.uci()
    pgn = board.san(result.move)
    print(f"UCI - Selected move from stockfish {uci}")
    print(f"PGN - Selected move from stockfish {pgn}")
    board.push(result.move)
    # target_move = board.san(result.move)
    send_moves(result.move.uci(), url)
    # return result.move.uci()
    return uci, pgn 