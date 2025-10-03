import csv, random
random.seed(7)

def gen_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(rows)

# Spam
spam=[]
for _ in range(150):
    free= random.randint(0,10); win= random.randint(0,10); click= random.randint(0,10)
    link= random.randint(0,1); subj = random.randint(5,120)
    score = 0.6*free + 0.7*win + 0.8*click + 1.2*link - 0.01*subj
    y = 1 if score>5 else 0
    spam.append([free,win,click,link,subj,y])
gen_csv("spam.csv", ["free_count","win_count","click_count","has_link","subject_len","spam"], spam)

# Clima (llueve/no)
clima=[]
for _ in range(150):
    import random as R
    temp = R.uniform(5,35); humedad = R.uniform(10,100)
    viento = R.uniform(0,40); presion = R.uniform(980,1030)
    score = 0.03*humedad - 0.04*(presion-1000) - 0.02*viento + 0.01*(25-temp)
    y = 1 if score>0.2 else 0
    clima.append([round(temp,2), round(humedad,2), round(viento,2), round(presion,1), y])
gen_csv("clima.csv", ["temp","humedad","viento","presion","llueve"], clima)

# Fraude
fraude=[]
for _ in range(150):
    import random as R
    monto = R.uniform(1,2000); online = R.randint(0,1)
    riesgo = R.randint(0,1); hora = R.randint(0,23)
    score = 0.002*monto + 0.9*online + 1.0*riesgo + (0.3 if (hora<6 or hora>22) else 0.0)
    y = 1 if score>1.5 else 0
    fraude.append([round(monto,2), online, riesgo, hora, y])
gen_csv("fraude.csv", ["monto","canal_online","pais_riesgo","hora","fraude"], fraude)

# Riesgo académico
riesgo=[]
for _ in range(150):
    import random as R
    promedio = R.uniform(0,20); ausencias = R.randint(0,30)
    tareas = R.randint(0,10); particip = R.randint(0,10)
    score = -0.15*promedio + 0.12*ausencias + 0.2*tareas - 0.05*particip
    y = 1 if score>0.5 else 0
    riesgo.append([round(promedio,2), ausencias, tareas, particip, y])
gen_csv("riesgo.csv", ["promedio","ausencias","tareas_pend","participacion","riesgo"], riesgo)

# AND / OR
gen_csv("and.csv", ["x1","x2","y"], [[0,0,0],[0,1,0],[1,0,0],[1,1,1]])
gen_csv("or.csv",  ["x1","x2","y"], [[0,0,0],[0,1,1],[1,0,1],[1,1,1]])
print("Listo: spam.csv, clima.csv, fraude.csv, riesgo.csv, and.csv, or.csv")
