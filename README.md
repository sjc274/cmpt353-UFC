
# Octagon analysis - Machine Learning With UFC Data (CMPT353 Project)


# Introduction ✏️

The Ultimate Fighting Championship (UFC) is the largest and most prominent mixed martial arts (MMA) organization in the world. Founded in 1993, the UFC has revolutionized the sport of MMA, bringing together athletes from various disciplines such as boxing, wrestling, Brazilian Jiu-Jitsu, Muay Thai, and others to compete in a regulated, professional environment.

This project is aim to build up machine learning models with UFC fighters' stature and fighting skill data, analyze the important features that effect the win or lose of each fight, and eventually predict the result of fights.
## Important Links 🔗

| [Dataset Download](https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset) | [Project report](https://docs.google.com/document/d/1YDo83rJPjpeeZwfCXiJiUb2g-wpk8kBKlAW805uosUw/edit?usp=sharing) 


- Dataset Download: Link to download the dataset of this project.
- Project report: Link to project report document.

## Table of Contents
1. [Demo](#demo)

2. [Installation](#installation)

3. [Reproducing this project](#repro)

### Layout Overview

```bash
repository
├── ipynb                       ## Jupyter notebook files with graph
├── src                         ## Source code of UFC data analysis
├── README.md                   ## Introduction of the project
├── requirements.txt            ## Setup requirements
```

<a name="demo"></a>
## Demo 📝

1) ### Data Cleaning

2) ### Machine Learning

3) ### Stature Data
- To analyze the feature importance: `python3 StatureDataAnalysis.py`
We made use of built-in feature `feature_importance_` to testify the feature importance:
```python
def ShowFeatureImportance(X, y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    model = model.fit(X_train, y_train)

    hist_df = pd.DataFrame({'Feature': X.columns, 'Feature importance': model.feature_importances_})
    hist_df = hist_df.sort_values(by='Feature importance', ascending=True)
    plt.figure(figsize=(10, 4))
    plt.barh(hist_df['Feature'], hist_df['Feature importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(model)
    plt.tight_layout()
    plt.show()
```


- To analyze evolution of UFC fighters' statures: `python3 StatureEvolutionAnalysis.py`
We reorgnized the data first, generated and plotted mean value throughout years, and applied normality test on 2010 and 2021 two years' data:
```python
for idx, feature in enumerate(['Height_cms', 'Reach_cms', 'Weight_lbs'], start=1):
    plt.subplot(2, 3, idx)
    data_2010 = fighters_2010[feature]
    mu_2010, std_2010 = data_2010.mean(), data_2010.std()
    xmin_2010, xmax_2010 = data_2010.min(), data_2010.max()
    x_2010 = np.linspace(xmin_2010, xmax_2010, 100)
    p_2010 = norm.pdf(x_2010, mu_2010, std_2010)
    plt.plot(x_2010, p_2010, 'k', linewidth=2)
    plt.hist(data_2010, bins=10, density=True, alpha=0.6, color='g')
    plt.title(f'Normal Distribution of {feature} (2010)')
    plt.xlabel(feature)
    plt.ylabel('Probability Density')
    plt.grid(True)
```

## 2. Installation
Please intall the requirement first: `pip install requirement.txt`
- To analyze the feature importance: `python3 StatureDataAnalysis.py`
- To analyze evolution of UFC fighters' statures: `python3 StatureEvolutionAnalysis.py`
<a name="repro"></a>
## 3. Reproduction
1) ### Data Cleaning

2) ### Machine Learning

3) ### Stature Data
Using the feature importance attribute of tree-based models, we generate the feature importance among three stature data (height, weight and reach) of fighters and plot them using DecisionTree model.
Here is the result graph:
![image](https://github.com/sjc274/cmpt353-UFC/assets/113268694/bb98277a-6f2a-405c-a99b-dcad4f9b6b94)

UFC fighters are increasing since 2010 and reached the peak in 2014. However their stature requirements are not as strict as it used to be before 2010. The average height weight and reach is decreasing year by year.
```terminal
	Height_cms_mean	Weight_lbs_mean	Reach_cms_mean	#Fighter_in_the_year
year				
2010	181.866008	181.916996	186.665929	253
2011	179.503294	171.082353	184.255059	339
2012	178.946667	168.251969	183.493281	381
2013	178.909095	167.513575	183.564842	442
2014	177.600429	162.100000	181.553589	559
2015	177.388592	162.507042	181.462606	568
2016	177.850800	164.300000	181.984909	550
2017	177.292000	162.669811	181.473245	530
2018	177.176140	160.915789	181.591702	568
2019	177.115000	160.805215	181.668712	594
2020	176.957520	162.108320	181.306028	580
2021	176.701261	160.253940	181.465429	551
```
We also did normality test based on each feature to justify if UFC fighters' builds are following the normal distribution.
Here is the result:
![image](https://github.com/sjc274/cmpt353-UFC/assets/113268694/a60eb9c2-66cf-4842-bf4b-de7c70fc4808)



## Data Downloading
Download the `ufc-master.csv`: https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset


