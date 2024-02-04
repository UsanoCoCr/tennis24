import random

def score_point(player_scores, player):
    """增加球员的得分，适用于常规游戏和抢七"""
    if player_scores[player] in ["0", "15", "30"]:
        player_scores[player] = str(int(player_scores[player]) + 15)
    elif player_scores[player] == "40":
        player_scores[player] = "Game"
    else:  # 抢七中的得分
        player_scores[player] += 1

def simulate_game(is_tiebreak=False):
    """模拟一局比赛，包括可能的抢七"""
    if not is_tiebreak:
        player_scores = {"player1": "0", "player2": "0"}
        while True:
            point_winner = random.choice(["player1", "player2"])
            score_point(player_scores, point_winner)
            
            # 检查是否有玩家赢得游戏
            if player_scores[point_winner] == "Game":
                return point_winner
            
            # 检查是否需要Deuce
            if player_scores["player1"] == "40" and player_scores["player2"] == "40":
                player_scores["player1"], player_scores["player2"] = "Deuce", "Deuce"
                
            # 处理Deuce情况
            if player_scores[point_winner] == "Deuce":
                player_scores[point_winner] = "Advantage"
                player_scores["player1" if point_winner == "player2" else "player2"] = "40"
            elif player_scores[point_winner] == "Advantage":
                player_scores[point_winner] = "Game"
    else:  # 抢七
        player_scores = {"player1": 0, "player2": 0}
        while True:
            point_winner = random.choice(["player1", "player2"])
            player_scores[point_winner] += 1
            
            # 检查是否有玩家赢得抢七
            if player_scores[point_winner] >= 7 and abs(player_scores["player1"] - player_scores["player2"]) >= 2:
                return point_winner

def simulate_set():
    """模拟一盘比赛，包括抢七"""
    set_score = {"player1": 0, "player2": 0}
    while True:
        if set_score["player1"] == 6 and set_score["player2"] == 6:
            # 进行抢七
            tiebreak_winner = simulate_game(is_tiebreak=True)
            set_score[tiebreak_winner] += 1
            return tiebreak_winner
        else:
            game_winner = simulate_game()
            set_score[game_winner] += 1
        
            # 检查是否有玩家赢得该盘
            if set_score[game_winner] >= 6 and abs(set_score["player1"] - set_score["player2"]) >= 2:
                return game_winner

def simulate_match(best_of=3):
    """模拟整场比赛，可能包括多盘"""
    match_score = {"player1": 0, "player2": 0}
    for _ in range(best_of):
        set_winner = simulate_set()
        match_score[set_winner] += 1
        if match_score[set_winner] > best_of // 2:
            return set_winner, match_score

# 模拟一场三盘两胜的比赛
winner, match_score = simulate_match(3)
print(f"Match winner: {winner} with score {match_score}")