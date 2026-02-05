
p_s = range(10)
q_s = range(10)
r_s = range(10)

pqr_list = []
i = 0
for p in p_s:
    for q in q_s:
        for r in r_s:
            value = 1 + p + q + r
            if value != 0 and 32 % value == 0:
                print(f"p={p}, q={q}, r={r} → 1+p+q+r={value} divides 32")
                print(i)
                i+=1
                pqr_list.append(f"{p}_{q}_{r}")

print(pqr_list)

print(pqr_list[0].split("_")[0])
print(pqr_list[0].split("_")[1])
print(pqr_list[0].split("_")[2])
