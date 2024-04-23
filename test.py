# import stockfish
# import chess
# import chess.engine
# import requests
# import chess.pgn

# board = chess.Board()
# engine = chess.engine.SimpleEngine.popen_uci("./stockfish/stockfish-ubuntu-x86-64-avx2")

# # testing for 3 move
# count = 0
# while not board.is_game_over():
#     result = engine.play(board, chess.engine.Limit(time=0.1))
#     print(f"Selected move from stockfish {result.move.uci()}")
#     print(f"Selected move from stockfish: {board.san(result.move)}")
#     board.push(result.move)
#     # test
#     count+=1
#     if count == 3:
#         break

def generate_prompt(moves:str):
    temp = moves.strip()
    temp_list = temp.split(" ")
    print(len(temp_list))
    prev = ' '.join(temp_list[:-1])
    if(len(temp_list) == 1):
        prev = "NONE"

    return f"In a paragraph, explain the rationale behind the last move, where all previous moves are - previous moves : {prev}, last move : {temp_list[-1]}."

print(generate_prompt("e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e4 e5 e4"))