from igraph import Graph # type: ignore
import numpy as np
import random

# ------------------ 확산 확률 추정기 ------------------
def estimate_spread_prob():
    print("=== 가짜 뉴스 확산 확률 추정기 ===\n")
    score = 0.1  # 기본값

    sensational = input("제목에 자극적 표현이 있나요? (y/n): ").lower() == 'y'
    has_source = input("신뢰할 만한 출처(언론사 등)가 있나요? (y/n): ").lower() == 'y'
    sender = input("누가 보냈나요? (지인/단톡방/모르는사람): ").lower()
    literacy = input("당신의 정보 판단력은? (높음/보통/낮음): ").lower()

    if sensational: score += 0.2
    if not has_source: score += 0.2
    if sender in ['단톡방', '모르는사람']: score += 0.1
    if literacy == '낮음': score += 0.2
    elif literacy == '보통': score += 0.1

    score = min(score, 1.0)
    print(f"\n추천 확산 확률 (가짜 뉴스): {score:.2f}")
    return score

# ------------------ 그래프 생성 ------------------
NUM_NODES = 1_000_000
EDGE_PROB = 0.00001
NUM_EDGES = int(NUM_NODES * (NUM_NODES - 1) * EDGE_PROB / 2)

print("\nGenerating large graph for numeric simulation...")
large_graph = Graph.Erdos_Renyi(n=NUM_NODES, m=NUM_EDGES, directed=False)

agent_types = np.random.choice(["high", "neutral", "low"], size=NUM_NODES, p=[0.3, 0.5, 0.2])

# ------------------ 시뮬레이션 함수 ------------------
def simulate_dual_spread(graph, agent_types, fake_prob_map, true_prob_map, seed_count=10, max_steps=20):
    infected_fake = np.zeros(NUM_NODES, dtype=bool)
    infected_true = np.zeros(NUM_NODES, dtype=bool)

    initial_seeds_fake = random.sample(range(NUM_NODES), seed_count)
    initial_seeds_true = random.sample(range(NUM_NODES), seed_count)

    infected_fake[initial_seeds_fake] = True
    infected_true[initial_seeds_true] = True

    current_fake = set(initial_seeds_fake)
    current_true = set(initial_seeds_true)

    history_fake = [np.sum(infected_fake)]
    history_true = [np.sum(infected_true)]

    fake_probs = np.array([fake_prob_map[t] for t in agent_types])
    true_probs = np.array([true_prob_map[t] for t in agent_types])

    for step in range(max_steps):
        new_fake = set()
        new_true = set()

        for node in current_fake:
            neighbors = graph.neighbors(node)
            for neighbor in neighbors:
                if not infected_fake[neighbor] and random.random() < fake_probs[neighbor]:
                    infected_fake[neighbor] = True
                    new_fake.add(neighbor)

        for node in current_true:
            neighbors = graph.neighbors(node)
            for neighbor in neighbors:
                if not infected_true[neighbor] and random.random() < true_probs[neighbor]:
                    infected_true[neighbor] = True
                    new_true.add(neighbor)

        if not new_fake and not new_true:
            break

        current_fake = new_fake
        current_true = new_true

        history_fake.append(np.sum(infected_fake))
        history_true.append(np.sum(infected_true))

    return history_fake, history_true

# ------------------ 실행 ------------------
print("\n--- 확산 확률 추정 후 수치 기반 시뮬레이션 실행 ---")
fake_prob = estimate_spread_prob()
true_prob = 0.1 

fake_prob_map = {"high": 0.1, "neutral": fake_prob, "low": 0.6}
true_prob_map = {"high": 0.05, "neutral": true_prob, "low": 0.2}

history_fake, history_true = simulate_dual_spread(
    large_graph,
    agent_types,
    fake_prob_map,
    true_prob_map,
    seed_count=10,
    max_steps=20
)

print("\n시간별 감염자 수 및 전체 대비 감염률:")
print("단위: 명 (퍼센트%)\n")
for step in range(len(history_fake)):
    count_fake = history_fake[step]
    count_true = history_true[step]
    rate_fake = count_fake / NUM_NODES * 100
    rate_true = count_true / NUM_NODES * 100
    print(f"Step {step}: 가짜 뉴스 = {count_fake:,}명 ({rate_fake:.4f}%), 일반 뉴스 = {count_true:,}명 ({rate_true:.4f}%)")
