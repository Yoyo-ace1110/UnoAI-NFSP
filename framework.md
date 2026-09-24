在基於 **NFSP (Neural Fictitious Self-Play)** 演算法實作 Uno AI 時，系統架構主要由四大關鍵元件組成：**遊戲環境（Game Rules/Engine）**、**Q-Network**、**Policy Network**，以及統籌一切的 **NFSP Agent**。

---

### 一、 遊戲規則/引擎（Game Engine & Environment）

遊戲引擎是 AI 的「物理定律與裁判」，負責維護對局狀態並提供互動介面。

#### 1. 狀態表徵（State Representation）

AI 無法直接看懂實體卡牌，必須將局勢轉換為數字張量（Tensor）。以標準 2 人 Uno 為例，狀態可編碼為 **$3 \times 4 \times 15$ 張量**：

* **通道 0（Channel 0）**：目前 AI 自己手上的牌。
* **通道 1（Channel 1）**：對手當前剩餘手牌張數與出牌歷史（隱藏資訊不可見，僅能紀錄歷史狀態）。
* **通道 2（Channel 2）**：場上最後打出的底牌（Top Card）與當前宣告的指定顏色。
* **維度說明**：$4$ 代表 4 種顏色（紅、黃、藍、綠），$15$ 代表 15 種牌型（0-9 數字牌、$+2$、Skip、Reverse、Wild、$+4$）。

#### 2. 動作空間與遮蔽（Action Space & Legal Action Masking）

* **動作空間（Action Space）**：預設 $61$ 種動作索引（包括打出特定顏色/數字的牌、變色卡選擇特定顏色打出、以及「從牌庫摸牌」）。
* **合法動作遮蔽（Legal Action Masking）**：引擎必須傳回一個 Boolean 向量（Mask）。例如手上只有紅牌，當場上是藍牌時，AI 只能選擇「摸牌」或打出「無顏色限制的 Wild 牌」。這個 Mask 能確保神經網路**絕不出錯牌（非法動作）**。

#### 3. 轉移與獎勵機制（State Transition & Reward）

* **轉移**：執行動作後，更新底牌、輪換回合，若打出 $+2$/$+4$ 則強制下家摸牌。
* **獎勵（Reward）**：出牌過程中預設為 $0$，直到遊戲結束：獲勝給 $+1$，失敗給 $-1$。

---

### 二、 Q-Network（動作價值網路 / 最佳回應）

Q-Network 的職責是**追求眼前勝率最大化**，尋找針對當前對手打法的「最佳攻擊路徑（Best Response）」。

* **輸入**：當前盤面狀態張量 $S$。
* **輸出**：所有 $61$ 種動作的預期勝率 $Q(S, A)$。
* **選牌邏輯**：加上合法動作遮蔽後，挑選 $Q$ 值最高的合法動作：

$$\text{Action} = \arg\max_{A \in \text{Legal}} Q(S, A)$$


* **記憶庫**：將所有對局經驗 $(S, A, R, S', \text{Done})$ 寫入 **Replay Buffer**。
* **訓練方式**：採用 **Double DQN** 演算法，從 Replay Buffer 隨機抽樣（Batch Sample），最小化 TD 均方誤差（MSE Loss）：

$$\text{Loss}_Q = \left( R + \gamma \cdot Q_{\text{target}}\left(S', \arg\max_{a} Q(S', a)\right) - Q(S, A) \right)^2$$



---

### 三、 Policy Network（策略網路 / 動態平均）

Policy Network 的職責是**防止打法過於好猜**。它透過監督式學習吸收歷史上所有成功經驗，輸出一個「平穩、機率化」的出牌策略。

* **輸入**：當前盤面狀態張量 $S$。
* **輸出**：所有 $61$ 種動作的**機率分佈** $P(A \mid S)$（通過 `Softmax` 激活函數）。
* **選牌邏輯**：經合法動作遮蔽過濾並重新歸一化後，依據機率採樣（Sampling）出牌，而非死板地選最大值。
* **記憶庫**：使用 **Reservoir Sampling Buffer**（水塘採樣記憶庫），僅儲存 **Q-Network 被選中並成功出牌**時的 $(S, A)$ 數據。
* **訓練方式**：採用**監督式交叉熵損失（Cross-Entropy Loss / Behavioral Cloning）**：

$$\text{Loss}_{\pi} = -\log P(A \mid S)$$



---

### 四、 Agent（NFSP 大腦與決策控制器）

Agent 是包裝了 Q-Network、Policy Network 以及記憶庫的控制器，對外與遊戲引擎互動，對內控制模型訓練。

```
                       ┌───────────────────────────────┐
                       │       遊戲環境 (Engine)        │
                       └───────────────┬───────────────┘
                                       │ 傳入 State (S)
                                       ▼
                       ┌───────────────────────────────┐
                       │          NFSP Agent           │
                       │   (依據參數 η 隨機切換模式)      │
                       └───────┬───────────────┬───────┘
  以機率 η (例如 10%) 選擇     │               │ 以機率 1-η (例如 90%) 選擇
                               ▼               ▼
                   ┌──────────────┐         ┌──────────────┐
                   │  Q-Network   │         │Policy Network│
                   │ (極限攻擊打法) │         │ (動態平穩打法) │
                   └───────┬──────┘         └───────┬──────┘
                           │                        │
                           └───────────┬────────────┘
                                       ▼
                             決定最終動作 (Action)

```

#### 1. 決策運作機制（$\eta$-Greedy Strategy Selection）

在每一手牌決策時，Agent 會有一個切換參數 $\eta$（例如 $0.1$）：

* **以機率 $\eta$（10%）**：使用 **Q-Network** 選牌。
* *目的*：不斷探索與嘗試新的進攻破局招式。
* *動作紀錄*：此時的 $(S, A)$ 會被同步寫入 Policy Network 的 Reservoir Memory 中，作為「優良戰術經驗」供策略網路學習。


* **以機率 $1-\eta$（90%）**：使用 **Policy Network** 選牌。
* *目的*：使用混淆對手的平均策略，維護賽局的納許均衡（Nash Equilibrium）。



#### 2. Agent 的內部訓練迴圈（Training Loop）

在每回合對局結束時，Agent 會自動呼叫訓練步驟：

1. **抽樣 Replay Buffer** $\rightarrow$ 梯度更新 **Q-Network**。
2. **抽樣 Reservoir Memory** $\rightarrow$ 梯度更新 **Policy Network**。

---

### 💡 專案架構彙整與程式碼組織

若要從頭建立專案，檔案架構建議如下規劃：

```text
uno_project/
├── engine/
│   ├── card.py          # 卡牌類別 (顏色、數字、功能)
│   ├── deck.py          # 發牌與牌庫管理
│   └── uno_env.py       # 遊戲主邏輯 (處理 State 張量化、Legal Mask、Reward)
├── models/
│   ├── q_network.py     # PyTorch Q-Network (輸入 3x4x15 Tensor -> 輸出 61 Q-values)
│   └── policy_network.py# PyTorch Policy Network (輸入 3x4x15 Tensor -> 輸出 61 Probabilities)
├── agents/
│   ├── memory.py        # ReplayBuffer & ReservoirBuffer
│   └── nfsp_agent.py    # 包含 η 切換決策、Loss 計算與更新邏輯
└── train_offline.py     # 離線預訓練 (使用 Dataset) 與 線上 Self-Play 訓練主程式

```
