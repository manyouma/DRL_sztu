#  Introduction to Reinforcement Learning

Welcome to **IB00398 Introduction to Reinforcement Learning** at Shenzhen Technology University!  

### Course News & Announcements
- **[2025-11-04]**  Part 2a: "Simplified Simulator" of the project is due on Nov. 12th. Please create a GitHub repository containing an OpenAI Gym–style environment and trajectories generated with at least two baseline algorithms, and share the link with me via email. <br>



<details>
<summary>&nbsp;&nbsp;&nbsp;&nbsp;Archived News</summary>
    - **[2025-10-15]** Template for the project report 1 posted. The due date is October 22nd. <br>
    - **[2025-10-12]** Please complete the [Academic Integrity and Intellectual Property Questionnaire](https://github.com/manyouma/DRL_sztu/blob/main/project_guide/IW_IP.pdf) by Oct 15.  <br>
    - **[2025-10-11]** All future lecture are moved to C-1-116 (BDI Building). <br>
    - **[2025-10-10]** There will be a make-up lecture on Oct. 11th (Saturday). <br>
    - **[2025-09-26]** Project 1a posted. The due date is Oct. 9th. <br>
    - **[2025-09-24]** Lab 2 solutions posted. <br>
    - **[2025-09-23]** Lecture on September 24 is cancelled due to the typhoon landing. Stay safe everyone! <br>
    - **[2025-09-18]** Lab 1 solutions posted. <br>
    - **[2025-09-17]** Course GitHub repo opened. <br>

</details>

### Course Information 

| **Item**          | **Details** |
|-------------------|-------------|
| **Instructor**    | Dr. Manyou Ma <br> Assistant Professor, School of Big Data and Internet, Shenzhen Technology University <br> Email: [mamanyou@sztu.edu.cn](mailto:mamanyou@sztu.edu.cn) <br> Office: Room 1717, C1 (BDI Building) <br> Office Hours: Monday 2:00 pm - 3:00 pm (email to schedule) |
| **Lectures**      | Wednesday, Period 6-7 (14:00–15:30) — C1 Room 116 (BDI Building)  <br> Thursday, Period 1-2 (08:30–10:00) — C1 Room 116 (BDI Building) |
| **Credits**       | 4 |
| **Prerequisites** | Python programming, basic probability & linear algebra |
| **Grading**       | Attendance: 5% ; Lab Reports: 15% ; Project Presentation: 20% ; Final Project: 60% <br>  All labs are due one week after they are posted. Late submissions will incur a penalty of 20% per week.|


### Course Material 
| Week | Lecture Topic | Lab / Programming Focus |
|--------|------------------|------------------------|
| 01 (Sep 08)| Basic Concepts of RL [slides](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/Lecture%20slides/slidesContinuouslyUpdated/L1-Basic%20concepts.pdf) |  Introduction to OpenAI Gym and NumPy [lab](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab01_frozenLake_intro.ipynb) [sol](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab01_frozenLake_intro_ans.ipynb)|
| 02 (Sep 15)| State Values and Bellman Equation [slides](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/Lecture%20slides/slidesContinuouslyUpdated/L2-Bellman%20equation.pdf) | State transition probability of FrozenLake [lab](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab02_frozenLake_MDP.ipynb) [sol](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab02_frozenLake_MDP_sol.ipynb)|
| 03 (Sep 22)| Bellman Optimality Equation [slides](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/Lecture%20slides/slidesContinuouslyUpdated/L3-Bellman%20optimality%20equation.pdf) | Optimal Policy for FrozenLake [lab](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab03_frozenLake_optimal.ipynb) [sol](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab03_frozenLake_optimal_sol.ipynb)|
| 04 (Sep 29)| Value Iteration and Policy Iteration [slides](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/Lecture%20slides/slidesContinuouslyUpdated/L4-Value%20iteration%20and%20policy%20iteration.pdf) | Value Iterations for FrozenLake [lab](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab04_frozenLake_VI_PI.ipynb) [sol](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab04_frozenLake_VI_PI_sol.ipynb)|
| 05 (Oct 06)| Monte Carlo Methods [slides](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/Lecture%20slides/slidesContinuouslyUpdated/L5-Monte%20Carlo%20methods.pdf) | Policy Iterations for FrozenLake|
| 06 (Oct 13)| Stochastic Approximations [slides](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/Lecture%20slides/slidesContinuouslyUpdated/L6-Stochastic%20approximation.pdf) | Stochastic Approximation [lab](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab06_StochasticApproximation.ipynb) [sol](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab06_StochasticApproximation_Sol.ipynb) |
| 07 (Oct 20)| TD methods I [slides](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/Lecture%20slides/slidesContinuouslyUpdated/L7-Temporal-Difference%20Learning.pdf)  | Frozen Lake Encore [lab](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab07_frozenLake_revisited.ipynb)  [sol](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab07_frozenLake_revisited_sol.ipynb)       |
| 08 (Oct 27)| TD methods II | Q-learning  on Cart-Pole [lab](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab08_cartpole.ipynb) [sol](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab08_cartpole_sol.ipynb)|
| 09 (Nov 3) | Value-based methods I [slides](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning/blob/main/Lecture%20slides/slidesContinuouslyUpdated/L8-Value%20function%20methods.pdf.pdf)| 	DQN on Atari [lab](https://github.com/manyouma/DRL_sztu/blob/main/labs/Lab9_atari_DQN.ipynb) |
| 10 (Nov 10)| Value-based methods II | DQN on Atari |
| 11 (Nov 17)| Policy-based methods I |  Vanilla Policy Gradient |
| 12 (Nov 24)| Policy-based methods II | Optimization with REINFORCE |
| 13 (Dec 01)| Actor-Critic Methods | A2C and DDPG  |
| 14 (Dec 08)| Trust Region Methods I | TRPO |
| 15 (Dec 15)| Trust Region Methods II | PPO |
| 16 (Dec 22)| LLM and RLHF  | LLM fine-tuning |
| 17 (Dec 29)| Project Presentation  | Project Workshop |
| 18 (Jan 05)| Project Presentation  | Project Workshop |

### Project Information
| Project Assignment | Instructions | Due Date | Weight |
|--------------------|-------------|----------|--------|
| **1. Real-life Problem as an MDP** | |  | **15** |
| &nbsp;&nbsp;1a: Questionnaire|  [pdf](https://github.com/manyouma/DRL_sztu/blob/main/project_guide/instr_project01a.pdf) [latex](https://github.com/manyouma/DRL_sztu/blob/main/project_guide/version01.tex) | Oct. 9  | 2 |
| &nbsp;&nbsp;1b: Revised Questionnaire | [IP Statement](https://github.com/manyouma/DRL_sztu/blob/main/project_guide/IW_IP.pdf) | Oct. 15 | 1 |
| &nbsp;&nbsp;1c: Report 1       | Template [pdf](https://github.com/manyouma/DRL_sztu/blob/main/project_guide/report1.pdf) [latex](https://github.com/manyouma/DRL_sztu/blob/main/project_guide/template.zip) | Oct. 22  | 12 |
| **2. Simplified Simulator**    | |  |  **10** |
| &nbsp;&nbsp;2a: Simplified Simulator | | Nov. 12   | 7 |
| &nbsp;&nbsp;2b: DQN Solution   | | Nov. 19  | 3 |
| **3. Complete Simulalor**      | |  | **10** |
| &nbsp;&nbsp;3a: DRL Solution 1 | | TBD   | 7 |
| &nbsp;&nbsp;3b: DRL Solution 2 | | TBD  | 3 |
| **4. Final Solution & Report** | |  | **25** |
| &nbsp;&nbsp;4a: Final Solution | | TBD  | 5  |
| &nbsp;&nbsp;4b: Final Report   | | TBD  | 20 |