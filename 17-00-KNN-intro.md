---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region slideshow={"slide_type": "slide"} -->
# $KNN$ calcul des $k$ plus proches voisins

algorithme supervisé, non-paramétrique, de clustering
<!-- #endregion -->

## les notations

<!-- #region -->
$\mathcal{X}$ est l'espace des $n$ observations d'entrée


----------


$x_{i, 1  \leq i \leq n} \in \mathcal{X}$ est une observation  
appelées aussi un descripteur, un prédicteur, une variable indépendante...  
elle contient les $d$ caractéristiques (features) décrivant les données d'entrée


----------


$\mathcal{Y}$ est l'espace des $n$ sorties


----------


$y_{i, 1  \leq i \leq n} \in \mathcal{Y}$ est une sortie  
appelée aussi variable dépendante, réponse, étiquettes (label)

----------
  
  
$\mathcal{F} \in \mathcal{Y}^\mathcal{X}$ est l'espace des fonctions de modélisation considérées pour faire nos prédictions  
appelée: l'espace des hypothèses  
(ici ce sont les modèles à base de **KNN**)

ainsi, différents modèles pour l'apprentissage donneront différents espaces d'hypothèse

Comment choisit-on cet espace des fonctions hypothèse ?  
grâce à l'expertise et de manière empirique


le choix de l'espace des hypothèses est fondamental:
- l'espace est trop simple $\Rightarrow$ on peut ne pas trouver de fonction correcte  
possibilité de sous-apprentissage
- l'espace est trop complexe (trop générique) $\Rightarrow$ on peut ne pas trouver de fonction correcte  
trop long à calculer  
possibilité de sur-apprentissage

----------
  
la tâche d'apprentissage:
- on suppose que les $y_i$ ont été calculés par une fonction $\mathcal{\phi}: \mathcal{X} \rightarrow \mathcal{Y}$  
c'est notre fonction cible et naturellement on ne la connait pas  
par contre on a des réalisations $(X_i, y_i)$ dont on veut en déduire cette fonction  
(fonc



- on cherche une fonction hypothèse $f \in \mathcal{F}$ qui approche **au mieux** $\mathcal{\phi}$
  
on a donc besoin de:
- quantifier la qualité d'une hypothèse donc d'une **fonction de coût**  
et de chercher une hypothèse optimale dans $\mathcal{F}$ au sens de la fonction de coût  
  
----------

- si la sortie est une classe $\rightarrow$ **classification**

- si la sortie est une valeur réelle $\rightarrow$ **régression**
<!-- #endregion -->

<!-- #region -->
**Problème mal-posé** (Science Direct)

https://www.sciencedirect.com/science/article/pii/S0165168420302723#:~:text=The%20inverse%20problem%20refers%20to,problem%20exists%20in%20many%20applications.

[Deep Learning Methods for Solving Linear Inverse
Problems: Research Directions and Paradigms](./deeplearning-solving-linear-inverse-problems_1-s2.0-S0165168420302723-am.pdf)

The inverse problem refers to using the results of actual observations to infer the values of the parameters that characterize the system and to estimate data that are not easily directly observed.

The inverse problem exists in many applications. In geophysics, the inverse problem is solved to detect mineral deposits such as underground oil based on the observations of an acoustic wave which is sent from the surface of the earth. In medical imaging, the inverse problem is solved to reconstruct an image of the internal structure of the human body based on the X-ray signal passing through the human body. In mechanical engineering, the inverse problem is solved to perform nondestructive testing by processing the scattered field on the surface, which avoids expensive and destructive evaluation. In imaging, the inverse problem is solved to recover images of high quality from the lossy image, for example, image denoising and image super-resolution (SR).

Mathematically, the inverse problem can be described as the estimation of hidden parameters of the model $m \in \mathbb{R}^N$ from the observed data $d \in \mathbb{R}^M,$ where $N$ (possibly infinite) is the number of model parameters and $M$ is the dimension of observed data.

A general description of the inverse problem is  
$d = \mathcal{A} (m)$ (1)  
where $\mathcal{A}$ is an operator (the forward operator) mapping the model space to the data space.

An inverse problem is well-posed if it satisfies the following three properties:
- *Existence:* For any data $d$, there exists an $m$ that satisfies (1)  
which means there exists a model that fits the observed data
- *Uniqueness:* For every $d$, if there are $m_1$ and $m_2$ that satisfy (1), then $m_1 = m_2$,  
which means the model that fits the observed data is unique.
- *Stability:* $\mathcal{A}$ is a continuous map  
which means small changes in the observed data $d$ will make small changes in the estimated model parameters $m$.


If any of the three properties does not hold, the inverse problem is ill-posed.

Établir un prédicteur en utilisant la minimisation du risque empirique (i.e. l'erreur sur le jeu de données) est un problème mal-posé. 
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## l'idée des KNN

   - trouver les $k$ observations les plus proches d'une observation   
   - utiliser ces voisins pour déduire la sortie (classe ou valeur) de l'observation
   
ce modèle est dit **non paramétrique** seul $k$ doit être fixé, ensuite il se base uniquement sur les données d'entraînement  
(i.e. aucun paramètre à déterminer...)
   
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## le problème

>étant donnés:
>- un ensemble de points $\mathcal{X}$ dans un espace métrique $E$
>- un entier $0 < k < n$
>- un point $x$
>
> trouver les $k$ points les plus proches de $x$  
> et les utiliser pour prédire l'appartenance à une classe (classification) ou une valeur (régression)


on doit donc calculer des distances entre des points de $E$ et $x$ (ici point = observation)  
d'où la nécessité d'avoir un espace métrique  
i.e. *un ensemble au sein duquel une notion de distance entre les éléments de l'ensemble est définie* (Wikipedia)

les descripteurs doivent être de type numérique (pour le calcul de la distance)  
<!-- #endregion -->

**et déjà des problèmes...**

**1) à cause de *la malédiction de la grande dimensionnalité***  
les descripteurs doivent être peu nombreux
- quand le nombre de descripteurs (la dimension de notre problème) augmente
- les mesures de distance deviennent de moins en moins discriminantes
- i.e. plus le nombre de descripteurs est grand et plus les valeurs des distances vont avoir tendance à être les mêmes pour les différents couples de points
- plus la dimension est grande plus il nous faut d'observations...

**2) à cause de plages de valeurs trop différentes dans nos données**  
le calcul de la distance euclidienne sera dominé par l'attribut qui a la plage de valeur la plus grande  
e.g. pour les attributs $A_1 \in [0, 1]$ et $A_2 \in [1000, 10000]$ $A_2$ va clairement dominer le calcul  
on doit pré-traiter les données pour leur *mise à l'échelle*  avant de pouvoir les utiliser  
(ici pour calculer la proximité entre deux observations)  
**mise à l'échelle:**  
- on redimensionne les intervalles des valeurs des attributs pour mettre leurs valeurs valeurs sur une échelle commune  
- par exemple $[0, 1]$ pour le min-max $\dfrac{x - min}{max - min}$  
(si données à peu près uniformément réparties sur l'intervalle  genre age)  
- ou encore par normalisation $\dfrac{x - mean}{std}$  
(si données mal réparties genre salaires)

(mais vous pouvez aussi essayer les deux approches et évaluer leur influence sur votre modèle)

**3) à cause des choix à faire ...**

Si l'observation est à égale distance de deux observations différentes:
- Quelle observation sera considérée dans les KNN voisins ?
- La première qui sera vue par l'algorithme ?
- Peut-être l'algorithme ne regardera même pas la seconde (il a déjà $k-1$ observations et en cherche une)
- Le résultat va dépendre de l'ordre dans lequel les observations sont présentées à l'algorithme
- On a donc un non-déterminisme (comme souvent en machine-learning)


**et des choix...**  
- choisir la distance euclidiennt ($\sum^n_{i=1} |x^1_i- x^2_i|$), Manhattan ($\sqrt{\sum^n_{i=1} (x^1_i- x^2_i)^2}$), ...
- choisir $k$

<!-- #region slideshow={"slide_type": "slide"} -->
## les algorithmes...

- on calcule toutes les distances entre le point $x$ et les points de l'espace $E$, on retient les $k$ plus petites (naïf et cher en temps de calcul)
- ou (identique) on calcule $k$ fois le plus proche voisin sans reconsidérer les points déjà vus (naïf et cher en temps de calcul)


- on base l'algorithme sur un partitionnement de l'espace, comme les arbres $kd$ (de dimension $k$) (voir https://fr.wikipedia.org/wiki/Arbre_kd)
- ...
<!-- #endregion -->

## la classification

$y$ est de type catégorie  
c'est une valeur dans une classe de valeurs  

- on utilise les $k$ observations les plus proches d'une nouvelle observation pour calculer $y$
- pour une classe, on retient la classe *la plus représentée* autour de l'observation (un vote)


## régression

<!-- #region -->
$y$ est de type quantitatif  
c'est une valeur réelle

- on utilise les $k$ observations les plus proches d'une nouvelle observation pour calculer $y$
- en faisant une somme pondérée

la sortie (numérique i.e. prédiction d'un réel) sur les $k$ plus proches voisins est:

   - $y_q = \dfrac{\sum_{i=1}^k w_i x_i}{\sum_{i=1}^k w_i}$
   - $w_i$ est une pondération
   - si on associe le poids $1/k$ pour les $k$ plus proches voisins et $0$ aux autres points  
     on retrouve la moyenne

   
autre pondération ?
   - on peut pondérer les voisins suivant leur proximité  
   par exemple, on peut choisir comme poids $1/d$, $d$ étant la distance au voisin considéré  
   (plus le voisin est proche plus il contribue à la prédiction)
   
   
on peut faire de même pour une classification et donner un poids au $k$ plus proches voisins suivant les distances

en conclusion sur les pondérations

   - on attribue un poids à toutes les observations de notre jeu de données d'apprentissage

   - notre prédiction peut se calculer en sommant sur toutes les observations pas simplement les $k$

   - $y_q = \dfrac{\sum_{i=1}^N w_i y_i}{\sum_{i=1}^N w_i}$

   - on peut aussi exprimer ce poids comme une distance

   - $y_q = \dfrac{\sum_{i=1}^N (1/distance(X_q, X_i)) y_i}{\sum_{i=1}^N (1/distance(X_q, X_i))}$  
   ($1/d$ parce qu'on est inversement proportionnel à la distance: plus on est proche mieux c'est)

----------

**notion de *noyau* de l'espace**

   - considérer une région à la place d'un nombre de voisins
   
   - un noyau est une fonction de la distance entre le point et tous les autres points de l'espace
   
   
$y_q = \dfrac{\sum_{i=1}^N kernel_\lambda(distance(X_i, X_q)) \; y_i}{\sum_{i=1}^N kernel_\lambda(distance(X_i, X_q))}$

   - le noyau détermine la manière dont les poids vont diminuer

   - le paramètre $\lambda$ détermine la région (la *bande passante*)

----------

**note**:
   - quand les données sont denses, KNN est bon
   - il reste très sensible au bruit dans les données
   - et surtout KNN n'est pas bon quand vous avez des trous dans vos données et qu'il vous faut les interpoler

----------

**Comment déterminer $k$ ?**

on est en apprentissage supervisé, on var faire une classique cross-validation
- prendre des ensemble d'apprentissage et des ensemble de test
- faire varier $k$
- calculer erreur d'apprentissage (biais) et l'erreur de généralisation (variance)
- et garder le meilleur compromis pour $k$ i.e. ni en sous-apprentissage ni en sur-apprentissage
<!-- #endregion -->

```python
[(i, j) for i in range(3) for j in range(4) if i < j]
```

END
