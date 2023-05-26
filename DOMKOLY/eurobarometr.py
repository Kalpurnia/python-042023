import pandas 
from scipy import stats
import matplotlib.pyplot as plt



countries=pandas.read_csv("countries.csv")
# print (countries)


# 1. INFLACE 
inflace = pandas.read_csv("ukol_02_a.csv")
#print (inflace)

#test normality rozložení 
#H0: Míra obav z inflace v evropských zemích v létě 2022 (resp.v zimě 22/23) má normální rozložení. ¨
# H1: Míra obav z inflace v evropských zemích v létě 2022 (resp. v zimě 22/23) nemá má normální rozložení.

normalII=stats.shapiro(inflace["97"])
inflace["97"].plot.kde()
#plt.show()
#print (normalII)
#ShapiroResult(statistic=0.9694532752037048, pvalue=0.33090925216674805),
# p-value>0,05 - nezamítám nulovou hypotézu (neprokázali jsme H1)

normal = stats.shapiro(inflace["98"])
#print(normal)
inflace["98"].plot.kde()
#plt.show()
#ShapiroResult(statistic=0.9803104996681213, pvalue=0.687289297580719)
# p-value >0,05 - nezamítám nulovou hypotézu (neprokázali jsme H1) 

# zřejmě mohu s daty pracovat , jako by měla normální rozložení (i když ten graf není úplně 
# přesvedčivý? )

# test 
# H0 - Obavy z inflace byly v létě 2022 stejné jako v zimě 2023. 
# H1 - Obavy z inflace se v létě 2022 a v zimě 2022/23 lišily. 

# o 2 různá pozorování jednoho případu - státu - volím párová test 
# nezamítla jsem hypotétu o tom, že se jedná o normální rozložení 
# mohu zkusit t-test 


compare = stats.ttest_rel(inflace["97"], inflace["98"])

#print(compare)
#TtestResult(statistic=3.868878598419143, pvalue=0.0003938172257904746, df=40)

#p-value t-testu <0,05 , tedy zamítám nulovou hypotézu - tedy obavy z inflace se mezi
# sledovanými obdobími liší 




# 2. DŮVĚRA VE STÁT 

#H0: Důvěra ve NG (resp.EU) má normální rozložení. 
#H1: Důvěra v NG (resp.EU) nemá mormální rozložení. 


trust = pandas.read_csv("ukol_02_b.csv")
#print (trust)


EU_yes = pandas.merge(countries, trust, on="Country")
EU_yes.to_csv("EU_yes.csv")
#print (EU_yes)

EU_yes["National Government Trust"].plot.kde()
#plt.show()
normalNG=stats.shapiro(EU_yes["National Government Trust"])
#print(normalNG)
#ShapiroResult(statistic=0.9438267350196838, pvalue=0.15140558779239655

#p-value>0.05 ,  nezamítám nulovou hypotézu,čili považuju rozložení NG za normální 

EU_yes["EU Trust"].plot.kde()
#plt.show()
normalEU=stats.shapiro(EU_yes["EU Trust"])
#print(normalEU)
#ShapiroResult(statistic=0.9735807180404663, pvalue=0.6981646418571472)    
# p-value>0.05 ,  nezamítám nulovou hypotézu, čili považuju rozložení EU  za normální 
#ShapiroResult(statistic=0.9735807180404663, pvalue=0.6981646418571472)    

#H0: Mezi důvěrou v NG a důvěrou v EU není souvislost. 
#H1: Mezi důvěrou v NG a důvěrou v EU  je souvislost.

# data mají normální rozložení , použiju Pearsonův korelační koeficient 

corr = stats.pearsonr(EU_yes["National Government Trust"], EU_yes["EU Trust"])
#print (corr)
#PearsonRResult(statistic=0.6097186340024556, pvalue=0.0007345896228823406)
# p- value <0,05 - zamítám nulovou hypotézu(tedy přijímám alternativní a myslím, 
# tedy je souvislost, pearson je dost vysoký - na poměry sociologických dat určitě 
# a má kladnou hodnotu, takže asi můžeme uzavřít, že důvěra v EU a ve vlastní 
# národní stát pozitivně koreluje  )


# 3. Důvěra ve stát a v euro 

# úkolem je otestovat, zda lidé ve státech platících EU důveřují EU více. 

#H0: Důvěra v EU se mezi státy eurozony a mimo eurozonu neliší.
#H1: Důvěra v EU je ve státech  eurozony vyšší než ve státech mimo eurozonu. 


EURO_yes = EU_yes[EU_yes["Euro"]==1]
EURO_no = EU_yes[EU_yes["Euro"]==0]

print (EURO_yes, EURO_no)


comparetrust = stats.ttest_ind(EURO_yes["EU Trust"],(EURO_no["EU Trust"]), alternative='greater')
print(comparetrust)
#test_indResult(statistic=-0.33471431258258433, pvalue=0.6296836583625585)
# p-value> 0,05, nezamítáme nulovou hypotézu, důvěra v EU je v eurozoně vyšší 