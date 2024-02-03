import pandas as pd

data = pd.read_csv("./data/Wimbledon_featured_matches.csv")

df = pd.DataFrame(data)

# 初始化游戏获胜者列
df['game_victor'] = None

# 遍历DataFrame以设置game_victor
start_index = 0  # 每一局开始的行索引
for i in range(1, len(df)):
    # 查找每局的结束点，即当前行为(0,0)的点
    if df.iloc[i]['p1_score'] == 0 and df.iloc[i]['p2_score'] == 0:
        # 判断上一局的获胜者
        prev_row = df.iloc[i-1]
        if prev_row['p1_score'] > prev_row['p2_score']:
            winner = 1
        else:
            winner = 2
        
        # 设置从start_index到i-1（包含）的所有行的game_victor
        df.loc[start_index:i-1, 'game_victor'] = winner
        
        # 更新下一局开始的行索引
        start_index = i

# 最后一局的处理
if df.iloc[-1]['p1_score'] != 0 or df.iloc[-1]['p2_score'] != 0:
    # 判断最后一局的获胜者
    if df.iloc[-1]['p1_score'] > df.iloc[-1]['p2_score']:
        winner = 1
    else:
        winner = 2
    df.loc[start_index:, 'game_victor'] = winner
    

# 保存
df.to_csv('./data/Wimbledon_featured_matches.csv', index=False)
