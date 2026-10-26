在基於 **NFSP (Neural Fictitious Self-Play)** 演算法實作 Uno AI 時, 系統架構主要由四大關鍵元件組成: 
**遊戲環境(Game Rules/Engine)**、**Q-Network**、**Policy Network**, 以及統籌一切的 **NFSP Agent**。

---

### 一、 遊戲規則/引擎(Game Engine & Environment)

遊戲引擎是 AI 的裁判, 負責維護對局狀態並提供互動介面。

#### 1. 狀態表徵(State Representation)

AI 無法直接看懂實體卡牌, 必須將局勢轉換為數字張量(Tensor)。以標準 2 人 Uno 為例, 狀態可編碼為 **$4 \times 15$ 張量**: 

* **通道 0 (Channel 0)**: 目前 AI 自己手上的牌。
* **通道 1 (Channel 1)**: 對手與牌庫當前剩餘手牌張數。
* **通道 2 (Channel 2)**: 出牌歷史(隱藏資訊不可見, 僅能紀錄歷史狀態)
* **維度說明**: $4$ 代表 4 種顏色(紅、黃、藍、綠), $15$ 代表 15 種牌型(0-9 數字牌、$+2$、Skip、Reverse、Wild、$+4$)。

#### 2. 動作空間與遮蔽(Action Space & Legal Action Masking)

* **動作空間(Action Space)**: 預設 $61$ 種動作索引(包括打出特定顏色/數字的牌、變色卡選擇特定顏色打出、以及「從牌庫摸牌」)。
* **合法動作遮蔽(Legal Action Masking)**: 引擎必須傳回一個 Boolean 向量(Mask)。例如手上只有紅牌, 當場上是藍牌時, AI 只能選擇「摸牌」或打出「無顏色限制的 Wild 牌」。這個 Mask 能確保神經網路**絕不出錯牌(非法動作)**。

#### 3. 轉移與獎勵機制(State Transition & Reward)

* **轉移**: 執行動作後, 更新底牌、輪換回合, 若打出 $+2$/$+4$ 則強制下家摸牌。
* **獎勵(Reward)**: 出牌過程中預設為 $0$, 直到遊戲結束: 獲勝給 $+1$, 失敗給 $-1$。

---

### 二、 Q-Network(動作價值網路 / 最佳回應)

Q-Network 的職責是**追求眼前勝率最大化**, 尋找針對當前對手打法的「最佳路徑(Best Response)」。

* **輸入**: 當前盤面狀態張量 $S$。
* **輸出**: 所有 $61$ 種動作的預期勝率 $Q(S, A)$。
* **選牌邏輯**: 加上合法動作遮蔽後, 挑選 $Q$ 值最高的合法動作: 

$$\text{Action} = \arg\max_{A \in \text{Legal}} Q(S, A)$$

* **記憶庫**: 將所有對局經驗 $(S, A, R, S', \text{Done})$ 寫入 **Replay Buffer**。
* **訓練方式**: 採用 **Double DQN** 演算法, 從 Replay Buffer 隨機抽樣(Batch Sample), 最小化 TD 均方誤差(MSE Loss): 
* **貝爾曼方程 (Bellman Equation)**: 
  * $R$ (Reward): 當下這個動作拿到的即時獎勵分數。
  * $\gamma$    : 當下的動作對未來牌局影響力的衰減因子, 界於 0~1 之間, 越高則 ai 會越有遠見。
  * $\arg\max_{a} Q(S', a)$: 在下一步的盤勢 $S'$ 中, 挑選出能讓 Q 值最高的那個動作 $a$。
  * $Q(S, A)$   : 神經網路當初在下這一步棋時預測可獲得的獎勵分數。
  * 平方誤差     : 保持誤差為正, 並放大較大的錯誤讓神經網路修正。

$$\text{Loss}_Q = \left( \underbrace{R + \gamma \cdot Q_{\text{target}}\left(S', \arg\max_{a} Q(S', a)\right)}_{\text{目標值 (現實)}} - \underbrace{Q(S, A)}_{\text{預測值}} \right)^2$$


---

### 三、 Policy Network(策略網路 / 動態平均)

Policy Network 的職責是**防止打法過於好猜**。它透過監督式學習吸收歷史上所有成功經驗, 輸出一個「平穩、機率化」的出牌策略。

* **輸入**: 當前盤面狀態張量 $S$。
* **輸出**: 所有 $61$ 種動作的**機率分佈** $P(A \mid S)$(通過 `Softmax` 激活函數)。
* **選牌邏輯**: 經合法動作遮蔽過濾並重新歸一化後, 依據機率採樣(Sampling)出牌, 而非死板地選最大值。
* **記憶庫**: 使用 **Reservoir Sampling Buffer**(蓄水池記憶庫), 僅儲存 **Q-Network 被選中並成功出牌**時的 $(S, A)$ 數據。
* **訓練方式**: 採用**監督式交叉熵損失(Cross-Entropy Loss / Behavioral Cloning)**: 
* **激活函數**: 使用ReLU, 公式: relu(x) = max(x, 0), 可避免神經元死亡。
* **$P(A \mid S)$**: 在當前狀態 $S$ 下, Policy Network 預測「出這張牌 $A$」的**機率**(介於 0 到 1 之間)。
* **損失函數**
  * $P(A \mid S)$: 在狀態 S 下, 策略網路選擇動作 A 的機率。
  * **負號($-$)**: 加上負號後, **機率越低, Loss 就越高；機率越高, Loss 越低**。
  * **$\log$**: 數學上用來把機率壓平並放大的工具,  如果機率很低, $\log$ 的值就會變成很大的負數。

$$\text{Loss}_{\pi} = -\log P(A \mid S)$$

---

### 四、 Agent(NFSP 大腦與決策控制器)

Agent 是包裝了 Q-Network、Policy Network 以及記憶庫的控制器, 對外與遊戲引擎互動, 對內控制模型訓練。

```
                       ┌───────────────────────────────┐
                       │       遊戲環境 (core)          │
                       └───────────────┬───────────────┘
                                       │ 傳入 State (S)
                                       ▼
                       ┌───────────────────────────────┐
                       │          NFSP Agent           │
                       │   (依據參數 η 隨機切換模式)     │
                       └───────┬───────────────┬───────┘
                  以機率 η 選擇 │               │ 以機率 1-η 選擇
                               ▼               ▼
                   ┌──────────────┐         ┌──────────────┐
                   │  Q-Network   │         │Policy Network│
                   │(追求勝率打法) │         │ (動態平穩打法)│
                   └───────┬──────┘         └───────┬──────┘
                           │                        │
                           └───────────┬────────────┘
                                       ▼
                             決定最終動作 (Action)

```

#### 1. 決策運作機制($\eta$-Greedy Strategy Selection)

在每一手牌決策時, Agent 會有一個切換參數 $\eta$(例如 $0.1$): 

* **以機率 $\eta$(10%)**: 使用 **Q-Network** 選牌。
* *目的*: 不斷探索與嘗試新的進攻破局招式。
* *動作紀錄*: 此時的 $(S, A)$ 會被同步寫入 Policy Network 的 Reservoir Memory 中, 作為「優良戰術經驗」供策略網路學習。


* **以機率 $1-\eta$(90%)**: 使用 **Policy Network** 選牌。
* *目的*: 使用混淆對手的平均策略, 維護賽局的納許均衡(Nash Equilibrium)。

#### 2. Agent 的內部訓練迴圈(Training Loop)

在每回合對局結束時, Agent 會自動呼叫訓練步驟: 

1. **抽樣 Replay Buffer** $\rightarrow$ 梯度更新 **Q-Network**。
2. **抽樣 Reservoir Memory** $\rightarrow$ 梯度更新 **Policy Network**。

---

### 💡 專案架構彙整與程式碼組織

```text
UnoAI-NFSP/
├── engine/
│   ├── player.py         # 玩家類別
│   └── uno_game.py       # 遊戲主迴圈
├── core/
│   ├── rele.py           # 遊戲規則
│   ├── card.py           # 卡牌類別
│   └── deck.py           # 發牌與牌庫管理
├── models/ 
│   ├── p_network.py      # PyTorch Policy Network && ReservoirBuffer (輸入 Tensor -> 輸出 Probabilities[61])
│   └── q_network.py      # PyTorch Q-Network      && ReplayBuffer    (輸入 Tensor -> 輸出 Quality-value[61])
├── agents/ 
│   ├── agent.py        # Agent父類別
│   ├── agent_v1.py     # AI出牌決策(v1)
│   ├── human_agent.py  # 人類玩家出牌決策
│   └── random_agent.py # 純隨機AI出牌決策
└── game/ 
    ├── forward.py          # 通用的前置作業
    ├── selfplay.py         # 自我訓練主程式
    ├── crossplay.py        # 交叉訓練主程式
    ├── humanplay.py        # 遊戲主程式
    └── ELO_ranking.py      # 評分機制
```
