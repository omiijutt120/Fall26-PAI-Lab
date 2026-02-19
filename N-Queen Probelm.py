def solve(n):
    pos = [-1] * n
    ans = []

    def safe(r, c):
        for i in range(r):
            if pos[i] == c or abs(pos[i] - c) == abs(i - r):
                return False
        return True

    def place(r):
        if r == n:
            ans.append(pos[:])
            return
        
        for c in range(n):
            if safe(r, c):
                pos[r] = c
                place(r + 1)
                pos[r] = -1

    place(0)
    return ans


n = int(input("Enter N: "))
res = solve(n)

print("\nTotal Solutions:", len(res), "\n")

for k in range(len(res)):
    print("Solution", k + 1)
    for r in res[k]:
        for i in range(n):
            if i == r:
                print("Q", end=" ")
            else:
                print(".", end=" ")
        print()
    print()
