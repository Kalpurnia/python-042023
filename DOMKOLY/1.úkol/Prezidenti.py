import pandas 
import  matplotlib.pyplot as plt
presidents = pandas.read_csv("1976-2020-president.csv")
#print(presidents.head(30))

# jen sloupečky, které jsou uvedeny jako důležité (year, state, party_simplified, candidatevotes, totalvotes) 
presidents = presidents.loc[:, ["year", "state", "party_simplified","candidatevotes", "totalvotes"]]
#print(presidents.head(30))

# seřazení podle roků a  + pořadí podle počtu hlasů v jednotlivých státech;
# předpokládám, že kdyby náhodou měli 2 stejný počet hlasů, budou mít stejné vyšší pořadi ? 

presidents["ranking"] = presidents.groupby(["state","year"])["candidatevotes"].rank(method="max", ascending = False)
#print(presidents.head(50))

# jen vítězové v každém roce dle států, dotazování:

winners_only = presidents[presidents["ranking"]==1]
two_major_competitors = presidents[(presidents["ranking"]==1) |(presidents["ranking"]==2)] 
#print (two_major_competitors)

#print (winners_only.head(30))
#print (winners_only.tail(30))

# nyní mám za každý stát jednoho vítěze (resp. stranu, která v daném roce zvítězila), ale 
# musím to seřadit podle států, ne podle let, abych mohla použít metodu shift . Řadit 
#se bude podle od nejstarších voleb po nejnovější a státy podle abecedy od A. 

winners_only=winners_only.sort_values(["state","year"], ascending=[True, True])
winners_switch = winners_only.copy()
two_major_competitors= two_major_competitors.sort_values(["state","year"], ascending=[True, True]) 
#print (winners_only.head(30))
#print (winners_only.tail(30))
#print (two_major_competitors.head(30))


# s pomocí metody shift()vytvořit nový sloupec s názvev vítezsné strany v následujícím roce

winners_only["next_election_winner"]=winners_only.groupby(["state"])["party_simplified"].shift(-1)

#print (winners_only.head(30))
#print (winners_only.tail(30))

# zkusím nadefinovat funkci, která bude vracet 0 (beze změny), 1(změna strany)
def party_change (row): 
    # for state in row.loc["state"]: (po konzultaci - zbytečný cyklus, metoda apply postupuje po řádcích)
        if pandas.isnull(row["next_election_winner"]):
            return None 
        elif row.loc["party_simplified"] == row.loc["next_election_winner"]:
            return 0
        else:
            return 1
        

winners_only["result_change"]=winners_only.apply(party_change, axis=1)      

#print (winners_only.head(30))
#print (winners_only.tail(30))

# teď by mělo jít pomocí cumsum zjistit, kdo měl nejvíc změn. 
winners_only_means =winners_only.groupby("state")["result_change"].mean(numeric_only=True)
#print(winners_only_means.sort_values())


winners_only = winners_only.dropna().reset_index()

# pro původní nejasnosti se sum() jsem použila cumsum() a protože je to kumiulativní, vyfiltrovala dotazováním  jsem pak jen údaj 
#za poslední rok, tedy 2016 (2020 vyhozen výše pomocí dropna())

winners_only["number_of_changes"]=winners_only.groupby(["state"])["result_change"].cumsum()
winners_only = winners_only[winners_only["year"]==2016]

# a teď tedy uspořádat - v prvé řadě podle počtu změn a protože hodně států má stejný 
# počet změn a tedy stejné pořadí, tak v rámci toho pořadí ještě státy podle abecedy 
winners_only = winners_only.sort_values(["number_of_changes", "state"],ascending=[True, True])


"""PO KONZULTACI - VYUŽITÍ SUM() by mělo být takto, ale už nechávám původní postup. 
### nicméně výhodám tohoto rozumím. 
winners_only_sum =winners_only.groupby("state")["result_change"].sum(numeric_only=True)
winners_only_sum =winners_only_sum.sort_values(ascending=True)
print (winners_only_sum)"""


# Graf 

# 10 států s nejvíce změnami 

ten_the_most_turbulent = winners_only.tail(10)
#print(ten_the_most_turbulent)

# překlasifikování "state" na index kvůli popisku osy y (i když mi stále není úplně jasné, proč se dole do příkazu 
# k popisklu dat nemůže napsat rovnou "state" - nefungovalo to) 
ten_the_most_turbulent = ten_the_most_turbulent.set_index("state")

ten_the_most_turbulent["number_of_changes"].plot(kind="bar")
plt.bar(ten_the_most_turbulent["index"], ten_the_most_turbulent["number_of_changes"] )
plt.ylabel("Počet změn politické strany")
plt.xlabel("Stát")
#plt.show()


###II. ČÁST - nahoře jsem si předpřipravila tu tabulku se dvěma hlavními rivaly (df two_major_competitors)two_major_competitors

two_major_competitors= two_major_competitors.sort_values(["state", "year", "candidatevotes"], ascending=[True, True, False])
two_major_competitors["compare_votes"]=two_major_competitors.groupby(["year", "state"])["candidatevotes"].shift(-1) 

#print(two_major_competitors)

two_major_competitors_diff = two_major_competitors.dropna().reset_index()
#print (two_major_competitors_diff)

two_major_competitors_diff["absolute_difference"]= two_major_competitors_diff["candidatevotes"] -two_major_competitors_diff["compare_votes"]
#print (two_major_competitors_diff)

#relativní margin 
 
two_major_competitors_diff["relative_margin"]=two_major_competitors_diff["absolute_difference"] /two_major_competitors_diff["totalvotes"]
#print(two_major_competitors_diff)


#print(two_major_competitors_diff.sort_values("relative_margin").head(1))

# pivot 
# udělala jsem si novou tabulku s jiným názvem jako kopii df winners_only (řádek 40)před tím, 
# než jsem ve winners only začala dělat další změny(např. filtrování roku 2016) -kopie se 
#jmenuje winners_switch
winners_switch["next_election_winner"]=winners_switch.groupby(["state"])["party_simplified"].shift(1)
winners_switch=winners_switch.dropna().reset_index()
print (winners_switch)
def party_comparison(row):
      if row["party_simplified"]==row["next_election_winner"]:
            return "no_change"
      if row["party_simplified"]in ["DEMOCRAT"] and row["next_election_winner"] in ["REPUBLICAN"]:
            return "from DEM to REP"
      if row["party_simplified"]in ["REPUBLICAN"] and row["next_election_winner"] in ["DEMOCRAT"]:
            return "from REP to DEM"
      else: 
            return None # (pro pořádek. Pravděpodobně nedošlo k tomu, že by někde na 
            # prvních dvou místech byla jiná strana, ale nekontrolovala jsem celou tabulku,
            #takže si nemůžu být jistá) 
      

winners_switch["switch"]=winners_switch.apply(party_comparison, axis=1)      
#print(winners_switch.head(50))


winners_switch_pivot_table = pandas.pivot_table(winners_switch, values = "state", index ="year", columns = "switch", aggfunc="count") 
print (winners_switch_pivot_table)