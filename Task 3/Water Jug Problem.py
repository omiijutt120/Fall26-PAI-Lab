cap1 = 4
cap2 = 3
goal = 2

visited = set()

def dfs(state, path):
    if state in visited:
        return None
    
    visited.add(state)
    path.append(state)
    
    x, y = state
    
    if x == goal or y == goal:
        return path
    
    next_states = []
    
    next_states.append((cap1, y))

    next_states.append((x, cap2))

    next_states.append((0, y))

    next_states.append((x, 0))

    pour = min(x, cap2 - y)
    next_states.append((x - pour, y + pour))

    pour = min(y, cap1 - x)
    next_states.append((x + pour, y - pour))
    
    pour = min(cap1, cap2 - y)
    next_states.append((cap1 - pour, y + pour))

    pour = min(cap2, cap1 - x)
    next_states.append((x + pour, cap2 - pour))
    
    for new_state in next_states:
        result = dfs(new_state, path.copy())
        if result:
            return result
    
    return None


start = (0, 0)
solution = dfs(start, [])

if solution:
    print("Solution found")
    for step, s in enumerate(solution):
        print(f"Step {step}: {s}")
else:
    print("No Solution")
