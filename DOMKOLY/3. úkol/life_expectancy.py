


import pandas 
import seaborn
import matplotlib.pyplot as plt
from scipy import stats 
import numpy 
import statsmodels.api as sm
import statsmodels.formula.api as smf


# I. ÚKOL 
lif_exp=pandas.read_csv("Life-Expectancy-Data-Updated.csv")

lif_exp=lif_exp[lif_exp["Year"]==2015]
#print(lif_exp)


# 1/ za celý soubor a bez transformace GDP: 
#seaborn.regplot(lif_exp, x = "GDP_per_capita", y ="Life_expectancy", scatter_kws={"s":3},line_kws={"color":"r"})
#plt.show()
formula = "Life_expectancy ~ GDP_per_capita"
mod=smf.ols(formula, lif_exp)
res=mod.fit()
#print(res.summary())
#R-squared:  0.396  (GDP vysvěluje necelých 40 % variability naděje na dožití)


#2/ přidat další proměnné do modelu: 

formula = "Life_expectancy ~ GDP_per_capita+ Schooling+Incidents_HIV+Diphtheria+Polio+BMI + Measles"
mod=smf.ols(formula, lif_exp)
res=mod.fit()
#print(res.summary())

# R-square(koeficitent determinace) - 0.790 (tedy tyto proměnné dohromady vysvětlují skoro 80 % variability
#  délky nadeje na dožití, tedy cca dvakrát tolik než v předchozím modelu samotné GDP )


"""GDP_per_capita  0.0001   1.96e-05      5.565      0.000    7.05e-05       0.000
Schooling          0.8445      0.146      5.791      0.000       0.557       1.132
Incidents_HIV     -1.4128      0.173     -8.154      0.000      -1.755      -1.071
Diphtheria        -0.0035      0.051     -0.067      0.946      -0.105       0.098
Polio              0.1385      0.060      2.304      0.022       0.020       0.257
BMI                0.4254      0.161      2.646      0.009       0.108       0.743
Measles            0.0390      0.023      1.731      0.085      -0.005       0.083"""

#největší váhu v toto modelu mají v kladném smyslu schooling a v negaticvním incidence HIV, tedy čím je v dané
# populaci vyšší výskyt HIV, tím je nižší naděje na dožití, čím více lidí absolvuje s školní docházku, tím je
# vyšší naděje na dožití lidí v rámci dané  populace 


# Normalita reziduí : 
# zvážení hypotézy normality reziduí: 
# hodnota Prob. příslušných  koeficientů (Omnibus, JB,) je větší než o,05 (konrétně 0.143, resp.0.138), tedy 
# nezamítáme nulovou hypotézu o normalitě rozložení reziduí (čili považujeme rozložení za normální)
# následně dle výše  P>|t| odebírám z modelu očkování proti záškrtu: 

formula = "Life_expectancy ~ GDP_per_capita+ Schooling+Incidents_HIV+Polio+BMI + Measles"
mod=smf.ols(formula, lif_exp)
res=mod.fit()
# print(res.summary())

# koeficent determinace se bez této proměnné nezměnil, vyšel opět 0.79






#-------------------------------------------------------------------------------------------------

#2/ Z hlediska statusu rozvojová/rozvinutá 

#print(lif_exp[["Country", "Economy_status_Developed","Economy_status_Developing" ]])

lif_exp_dev =lif_exp[lif_exp["Economy_status_Developed"]==1]
lif_exp_dev.to_csv("lif_exp_dev.csv")

lif_exp_underdev =lif_exp[lif_exp["Economy_status_Developed"]==0]
lif_exp_underdev.to_csv("lif_exp_underdev.csv")

#print(f"počet rozvinutých zemí : {lif_exp_dev.shape[0]}, počet rozvojových zemí:{lif_exp_underdev.shape[0]}")
# počet rozvinutých zemí : 37, počet rozvojových zemí:142

# ROZVINUTÉ
#seaborn.regplot(lif_exp_dev, x = "GDP_per_capita", y ="Life_expectancy", scatter_kws={"s":5},line_kws={"color":"r"})
# plt.show()
"""podstatně větší linearita u rozvinutých zemí než u celého souboru, ukazuje se,  
# že i při velikém rozdílu mezi minimem a maximem GDP (okometricky jen z grau, do cca 10 000 po více
# než 100 000 naděje na dožití  v zemích klasifikovaných jako rozvinuté osciluje v rozmezí 10 let, jinak 
# řečeno, jsou tam i země, kterým se navzdory malému HDP na hlavu podažilo srazit dětskou 
# úmrtnost na minimum - protože je to dětská úmrtnost, co snižuje celkovou naději na dožití;   )
zároveň si nemyslím, že by mělo smysl tady dělat logaritmickou transformaci dat, spíš si myslím, 
že by mělo smysl zvolit robustní regresi, která by těm outliers dala menší váhu"""

data_x = lif_exp_dev[["GDP_per_capita"]]
data_x=sm.add_constant(data_x)
mod = sm.RLM(lif_exp_dev["Life_expectancy"],data_x)
res = mod.fit()
#print(res.summary())

"""==================================================================================
                     coef    std err          z      P>|z|      [0.025      0.975]
----------------------------------------------------------------------------------
const             77.6785      0.762    101.878      0.000      76.184      79.173
GDP_per_capita  7.232e-05    1.8e-05      4.026      0.000    3.71e-05       0.000
=================================================================================="""
 
#print(sum(numpy.abs(res.resid)))
# součet reziduálů: 63.08745128444282


# když přidám BMI a konzumaci alkoholu:
data_x = lif_exp_dev[["GDP_per_capita", "BMI", "Alcohol_consumption"]]
data_x=sm.add_constant(data_x)
mod = sm.RLM(lif_exp_dev["Life_expectancy"],data_x)
res = mod.fit()
#print(res.summary())
#print(sum(numpy.abs(res.resid)))
#součet reziduálů:52.48184661534286 (nižší hodnota než předchozí, tedy lepší model)


# ROZVOJOVÉ: 
seaborn.regplot(lif_exp_underdev, x = "GDP_per_capita", y ="Life_expectancy", scatter_kws={"s":3},line_kws={"color":"r"})
# plt.show()
# tady je zajímavý vidět, že jsou anomálie - země klasifikované jako rozvojové, ale s vysokým 
# HDP na hlavu 

#lif_exp_underdev["Life_expectancy_square"]=numpy.square(lif_exp_underdev["Life_expectancy"])

#seaborn.regplot(lif_exp_underdev, x = "GDP_per_capita", y ="Life_expectancy_square", scatter_kws={"s":3},line_kws={"color":"r"})
# plt.show()

lif_exp_underdev["GDP_per_capita_log"]=numpy.log(lif_exp_underdev["GDP_per_capita"])
#seaborn.regplot(lif_exp_underdev, x = "GDP_per_capita_log", y ="Life_expectancy_square", scatter_kws={"s":3},line_kws={"color":"r"})
#plt.show()


formula = "Life_expectancy ~ GDP_per_capita"
mod=smf.ols(formula, lif_exp_underdev)
res=mod.fit()
print(res.summary())
# R-squared: 0.247


formula = "Life_expectancy ~ GDP_per_capita_log"
mod=smf.ols(formula, lif_exp_underdev)
res=mod.fit()
print(res.summary())
#R-squared: 0.521


#formula = "Life_expectancy_square ~ GDP_per_capita_log"
mod=smf.ols(formula, lif_exp_underdev)
res=mod.fit()
print(res.summary())
# R-squared: 0.537 (nejlepší model s oběma proměnnými transformovanými)

# ještě zkusím výše navržený model jen na rozvojové země: 
formula = "Life_expectancy ~ GDP_per_capita+ Schooling+Incidents_HIV+Diphtheria+Polio+BMI + Measles"
mod=smf.ols(formula, lif_exp_underdev)
res=mod.fit()
#print(res.summary())
#R-squared: 0.721 (čili nižší než za celý soubor ), nulovou hyp. o normalitě reziduí také nezamítáme""" 


 #s transformovaným GDP 
formula = "Life_expectancy ~ GDP_per_capita_log+ Schooling+Incidents_HIV+Diphtheria+Polio+BMI + Measles"
mod=smf.ols(formula, lif_exp_underdev)
res=mod.fit()
#print(res.summary())
# R-squared:0.764 (čili nepatrně lepší model než ten předchozí)

# a na základě  P>|t| mohu takřka beztrestně odebrat záškrt a spalničky: 
formula = "Life_expectancy ~ GDP_per_capita_log+ Schooling+Incidents_HIV+Polio+BMI"
mod=smf.ols(formula, lif_exp_underdev)
res=mod.fit()
#print(res.summary())
# R-squared: 0.763 """


# ještě zkusím odstranit outliers: 
 
lif_exp_underdev["Life_expectancyZscore"]=numpy.abs(stats.zscore(lif_exp_underdev["Life_expectancy"]))
lif_exp_underdev_out = lif_exp_underdev[lif_exp_underdev["Life_expectancyZscore"]<3]

lif_exp_underdev["GDP_per_capitaZscore"]=numpy.abs(stats.zscore(lif_exp_underdev["GDP_per_capita"]))
lif_exp_underdev_out= lif_exp_underdev[lif_exp_underdev["GDP_per_capitaZscore"]<3]

# ve výsledku se podařilo odstranit jen 3 případy : 

#seaborn.regplot(lif_exp_underdev_out, x = "GDP_per_capita", y ="Life_expectancy", scatter_kws={"s":3},line_kws={"color":"r"})
#plt.show()

formula = "Life_expectancy ~ GDP_per_capita+ Schooling+Incidents_HIV+Diphtheria+Polio+BMI + Measles"
mod=smf.ols(formula, lif_exp_underdev_out)
res=mod.fit()
#print(res.summary())
# R-squared: 0.712 (čili odstraněním 3 outliers se mi na vržený model zhoršil i oproti tomu s netransformovaným GDP, tomu 
# nerozumím. )

print (lif_exp["Thinness_five_nine_years"])