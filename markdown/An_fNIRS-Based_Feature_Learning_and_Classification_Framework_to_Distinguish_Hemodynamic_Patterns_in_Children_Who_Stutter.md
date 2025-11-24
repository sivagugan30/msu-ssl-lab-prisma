# An_fNIRS-Based_Feature_Learning_and_Classification_Framework_to_Distinguish_Hemodynamic_Patterns_in_Children_Who_Stutter

```
1254

IEEE TRANSACTIONS ON NEURAL SYSTEMS AND REHABILITATION ENGINEERING, VOL. 26, NO. 6, JUNE 2018

An fNIRS-Based Feature Learning and
Classiﬁcation Framework to Distinguish
Hemodynamic Patterns in Children Who Stutter

Rahilsadat Hosseini

, Bridget Walsh, Fenghua Tian, and Shouyi Wang

Abstract— Stuttering is a communication disorder that
affects approximately 1% of
the population. Although
5–8% of preschool children begin to stutter, the majority
will recover with or without intervention. There is a sig-
niﬁcant gap, however, in our understanding of why many
children recover from stuttering while others persist and
stutter throughout their lives. Detecting neurophysiological
biomarkers of stuttering persistence is a critical objec-
tive of this paper. In this paper, we developed a novel
supervised sparse feature learning approach to discover
discriminative biomarkers from functional near infrared
spectroscopy (fNIRS) brain imaging data recorded during
a speech production experiment from 46 children in three
groups: children who stutter (n = 16); children who do not
stutter (n = 16); and children who recovered from stuttering
(n = 14). We made an extensive feature analysis of the
cerebral hemodynamics from fNIRS signals and selected a
small number of important discriminative features using the
proposed sparse feature learning framework. The selected
features are capable of differentiating neural activation pat-
terns between children who do and do not stutter with an
accuracy of 87.5% based on a ﬁve-fold cross-validation
procedure. The discovered set cerebral hemodynamics fea-
tures are presented as a set of promising biomarkers to
elucidate the underlying neurophysiology in children who
have recovered or persisted in stuttering and to facilitate
future data-driven diagnostics in these children.

Index Terms— Stuttering, functional near-infrared spec-
troscopy (fNIRS), speech production, children, data mining,
feature extraction and selection, biomarkers, mutual infor-
mation, sparse modeling.

I. INTRODUCTION

S TUTTERING is a communication disorder character-

ized by involuntary disruptions in the forward ﬂow
of speech. These disruptions, referred to as stuttering-like

Manuscript received November 12, 2017; revised March 6, 2018;
accepted April 8, 2018. Date of publication April 20, 2018; date of current
version June 6, 2018. The work of B. Walsh was supported by NIH under
Grant NIH/NIDCD R03 DC013402 [13]. (Corresponding author:Shouyi
Wang.)

R. Hosseini and S. Wang are with the Department of Industrial, Manu-
facturing, and Systems Engineering, The University of Texas at Arlington,
Arlington, TX 76019 USA (e-mail: rahilsadat.hosseini@mavs.uta.edu;
shouyiw@uta.edu).

B. Walsh is with the Purdue University College of Health and Human
Sciences, West Lafayette, IN 47907 USA (e-mail: walshb16@msu.edu).
F. Tian is with the Department of Bioengineering, The Uni-
versity of Texas at Arlington, Arlington, TX 76010 USA (e-mail:
fenghua.tian@uta.edu).

Digital Object Identiﬁer 10.1109/TNSRE.2018.2829083

are

repetitions of

recognized as

disﬂuencies,
speech
sounds or syllables, blocks where no sound or breath
emerge, or prolongation of speech sounds. In recent years,
there has been considerable progress toward understanding the
origins of a historically enigmatic disorder. Past theories of
stuttering attempted to isolate speciﬁc factors such as anxiety,
linguistic planning deﬁciencies, or muscle hyperactivity as the
root cause of stuttering (for review, see [1]). More recently,
however, stuttering is hypothesized to be a multifactorial
disorder. Atypical development of the neural circuitry under-
lying speech production may adversely impact the different
cognitive, motor, linguistic, and emotional processes required
for ﬂuent speech production [2], [3].

The average age of stuttering onset

is 33 months [4].
Although, 5-8 %, of preschool children begin to stutter,
the majority (70-80 %) will recover with or without interven-
tion [4], [5]. Given the high probability of recovery, parents
often elect to postpone therapy to see if their child’s stuttering
resolves. However, delaying therapy in children at greater
risk for persistence allows maladaptive neural motor networks
to form that are challenging to treat in the future [4], [6].
The lifelong implications of stuttering are signiﬁcant, impact-
ing psychosocial development, education, and employment
achievement [7]–[10].

There is a signiﬁcant gap in our understanding of why
so many children recover while others persist in stuttering.
Established behavioral risk factors for stuttering persistence
include one or more of the following: positive family history,
later age of onset (i.e. stuttering began after 36 months), time
since onset, sex–boys are more likely to persist, and type
and frequency of disﬂuencies [4]. Combining behavioral risk
factors with objective, physiological biomarkers of stuttering
may constitute a more powerful approach to help identify
children at greater risk for chronic stuttering. Detecting such
physiological biomarkers of stuttering persistence is a critical
objective of our research [11], [12].

In our earlier study, Walsh et al. (2017) [13] recorded
cortical activity during overt speech production from children
who stutter and their ﬂuent peers. During the experiment,
the children completed a picture description task while we
recorded hemodynamic responses over neural regions involved
in speech production and implicated in the pathophysiology
of stuttering including:
inferior frontal gyrus (IFG), pre-
motor cortex (PMC), and superior temporal gyrus (STG)

1534-4320 © 2018 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission.
See http://www.ieee.org/publications_standards/publications/rights/index.html for more information.

Authorized licensed use limited to: Michigan State University. Downloaded on November 10,2025 at 21:05:52 UTC from IEEE Xplore.  Restrictions apply. 

HOSSEINI etal.: FNIRS-BASED FEATURE LEARNING AND CLASSIFICATION FRAMEWORK

1255

with functional near-infrared spectroscopy (fNIRS), which
is a safe, non-invasive optical neuroimaging technology that
relies upon neurovascular coupling to indirectly measure brain
activity. This is accomplished using near-infrared light
to
measure the relative changes in both oxygenated (Oxy-Hb)
and deoxygenated hemoglobin (Deoxy-Hb), two absorbing
chromophores in cerebral capillary blood [14]. fNIRS offers
signiﬁcant advantages including its relatively low cost and
greater tolerance for movement, making it a more child-
friendly neuroimaging approach. fNIRS has been used to
assess the regional activation, timing, and lateralization of cor-
tical activation for a diverse number of perceptual, language,
motor, and cognitive investigations (for review, [15]).

Using fNIRS to assess cortical activation during overt
speech production, we found markedly different speech-
evoked hemodynamic responses between the two groups of
children during ﬂuent speech production [13]. Whereas con-
trols showed clear activation over left dorsal IFG and left
PMC, characterized by increases in Oxy-Hb and decreases
in Deoxy-Hb, the children who stutter demonstrated deac-
tivation, or the reverse response over these left hemisphere
regions. The distinctions in hemodynamic patterns between
the groups may indicate dysfunctional organization of speech
planning and production processes associated with stuttering
and could represent potential biomarkers of stuttering.

Although different brain signal patterns can be observed
for stuttering and control group in our previous studies,
there is still a lack of reliable quantitative tools to evaluate
stuttering treatment and recovery process based on brain
activity patterns. In our previous studies, we have extensive
research efforts on specialized machine learning (ML) and
pattern recognition techniques for multivariate spatiotempo-
ral brain activity pattern identiﬁcation under different brain
states [16]–[19]. In this study, we aimed to detect neuro-
physiological biomarkers of stuttering using advanced ML
techniques. In particular, we performed ML models for two
experiments. In experiment (1), we made an extensive feature
extraction from fNIRS brain imaging data of 16 children who
stutter and 16 children in a control group collected in our pre-
vious study [13]. Next, we developed a novel supervised sparse
feature learning approach to discover a set of discriminative
biomarkers from a large set of fNIRS features, and construct
a classiﬁcation model to differentiate hemodynamic patterns
from children who do and do not stutter. In experiment (2),
we applied the constructed classiﬁcation model on a novel
test set of fNIRS data collected from a group of children who
had recovered from stuttering and underwent the same picture
description experiment. Using the novel test set with children’s
data that was not used to develop the initial algorithms allowed
us to assess the model generalization with the discovered
biomarkers from experiments (1) to (2). We elected to include
children who had recovered from stuttering in the test group
for theoretical and clinical bearings. Young children who begin
to stutter are far more likely to recover than persist. It is impor-
tant to assess the underlying neurophysiology of different
stuttering phenotypes to learn, for example, whether recovered
children’s hemodynamic patterns would classify them with the
group of controls or with the group of stuttering children.

Fig. 1. Approximate positions of emitters (orange circles) and detectors
(purple circles) are shown on a standard brain atlas (ICBM 152). The
probes were placed symmetrically over the left and right hemisphere, with
channels 1-5 spanning inferior frontal gyrus, channels 6-7 over superior
temporal gyrus, and channels 8-9 over precentral gyrus/premotor cortex.

These proof-of-concept experiments represent a critical step
toward identifying greater risk for persistence in younger
children near the onset of stuttering.

The remainder of the paper is organized as follows: In
Section 2, we present the methodology, including participant
and data collection details, fNIRS data feature extraction
and structured sparse feature selection models. In Section 3,
we present the results of the pattern discovery of biomarkers as
well as performance consistency on the novel test-set of data
from recovered children. In section 4, we discuss the selected
features and their interpretations in terms of brain regions of
interest. Finally, we conclude the study in section 5.

II. METHOD

A. Participants,fNIRSDataCollection&Pre-Processing

In experiment (1), fNIRS data from the 32 children who
participated in the Walsh et al. (2017) study [13] was analyzed;
16 children who stutter (13 males) and 16 age- and socioe-
conomic status-matched controls (11 males). The participants
were between the ages of 7-11 years (M = 9 years). Stuttering
diagnosis and exclusionary criteria are provided in [13].

In experiment (2), a group of 14 children (10 males)
between the ages of 8-16 years (M = 12 years) who recovered
from stuttering was analyzed as an additional test group. All
of the children completed a picture description experiment in
which they described aloud different picture scenes (talkžtri-
als) that randomly alternated with nullž trials in which they
watched a ﬁxation point on the monitor. In order to compare
hemodynamic responses among the groups of children, only
ﬂuent speech trials were considered in the analyses.

For each experiment, we recorded hemodynamic responses
with a continuous wave system (CW6; TechEn, Inc.) that
uses near-infrared lasers at 690 and 830 nm as light sources,
and avalanche photodiodes (APDs) as detectors for measuring
intensity changes in the diffused light at a 25-Hz sampling
rate. Each source/detector pair is referred to as a channel.
The fNIRS system acquired signals from 18 channels (9 over
the left hemisphere and 9 over homologous right hemisphere
regions) that were placed over ROIs relying on 10-20 system
coordinates Figure (1).

Data analysis is detailed in Walsh et al. [13]. Brieﬂy,
the fNIRS data was preprocessed using Homer2 software [20].
Usable channels of raw data were low-pass ﬁltered at 0.5 Hz
and high-pass ﬁltered at 0.03 Hz. Concentration changes in

Authorized licensed use limited to: Michigan State University. Downloaded on November 10,2025 at 21:05:52 UTC from IEEE Xplore.  Restrictions apply. 

1256

IEEE TRANSACTIONS ON NEURAL SYSTEMS AND REHABILITATION ENGINEERING, VOL. 26, NO. 6, JUNE 2018

• Morphological features comprised the number of peaks

and zero crossings and measures of curve length.

Fig. 2. Oxy-Hb hemodynamic responses averaged over all 18 channels
for each subject. Controls are plotted on the left (cyan curves) and stut-
terers on the right (magenta curves). The grand average hemodynamic
response across all channels and subjects is represented by the black
dashed curve.

Fig. 3. Deoxy-Hb hemodynamic responses averaged over all 18 chan-
nels for each subject. Controls are plotted on the left (cyan curves) and
children who stutter on the right (magenta curves). The grand average
hemodynamic response across all channels and subjects is represented
by the black dashed curve.

Oxy-Hb and Deoxy-Hb were then calculated and a correlation-
based signal improvement approach applied to the concentra-
tion data to reduce motion artifacts [21]. Finally, we derived
each child’s Oxy-Hb and Deoxy-Hb event-related hemody-
namic responses from all channels from stimulus onset to the
end of the trial. We then subtracted the average hemodynamic
response associated with the null trials from the average hemo-
dynamic response from the talk trials to derive a differential
hemodynamic response for each channel [22]. The average
Oxy-Hb and Deoxy-Hb hemodynamic response averaged over
all 18 channels is plotted as a function of time for each child
in Figure (2) and (3).

B. FeatureExtraction

As shown in Figure (4), each experimental

trial was
partitioned into three phases: perception or the see-phase
(0-2s, the children saw a picture on the monitor), the talk-
phase (3-8s, the children described aloud the picture), and the
recovery-phase (9-23s, the hemodynamic response returned
to baseline) for measurements of Oxy-Hb and Deoxy-Hb.
We extracted 21 features from each channel; 21 = 4 + 3 +
3 + 1 + (5 × 2(for 1 and 2 sec of delay)). These delays were
implemented to account for correlation of the signal to its
lagged values. The names of the feature group and subgroups
are shown in Figure (4). Therefore, for each subject with
18 channels of fNIRS data, there were 378 extracted features
from Oxy-Hb and Deoxy-hb measurements in each phase.

The extracted groups of features are summarized in the

following.

• Statistical features capture descriptive information of the

signals.

expressed

parameters

capture
as

• Hjorth
variation
signal
over
time
and
activity, mobility,
as:
deﬁned
complexity.. The
are
var(y(t )d y/dt
acti vi t y = V ar (y(t)), mobili t y =
,
var(y(t ))
complexi t y = mobilit y(y(t )d y/dt )

features

three

(cid:2)

.

mobilit y(y(t ))

• Normalized Area Under the Signal (NAUS) calculates the
sum of values which have been subtracted from a deﬁned
baseline divided by the sum of the absolute values for the
fNIRS signal.

• Autocorrelation captured the linear relationship of the
signal with its historical values considering 1 and 2 s
delays Kendall, partial, Spearman and Pearson are four
ways to compute autocorrelation.

• Bicorrelation computes the bicorrelation on the time
series Xv for given delays in τv . Bicorrelation is an exten-
sion of the autocorrelation to the third order moments,
where the two delays are selected so that the second delay
is twice the original, (i.e. x(t)x(t − τ )x(t − 2τ )). Given
a delay of τ and the standardized time series Xv with
length n, denoted as Yv , the bi corr (τ ) can be calculated
as:

(cid:3)

(1)

n−2τ
j =1 Yv ( j )Yv(τ + j )Yv(2τ + j )
n − (2 × τ )
1) PersonalizedFeatureNormalization: As illustrated in Fig-
ures (2) and (3) fNIRS signals vary dynamically across sub-
jects, imposing a challenge to biomedical research. Because
of inter-individual variability in signal features, it is difﬁcult
to build a robust diagnostic model to accurately discriminate
between groups of participants. Outliers can further distort
the trained model, thus impeding generalization. To tackle
these issues, we applied a personalized feature normalization
approach to standardize the extracted feature values of each
subject onto the same scale to enhance feature interpretability
across subjects.

To accomplish this, we calculated the upper and lower
limits for each extracted feature using the formula Vl=
max(minimum feature value, lower quartile + 1.5 × interquar-
tile range) for the lower limit, and Vu= min(maximum feature
value, upper quartile + 1.5 × interquartile range) for the upper
limit. Feature values outside of this deﬁned interval were
considered to be outliers and mapped to 0 or 1. More details
can be found in study [23]. Assuming the raw feature value
was Fraw, the scaled feature value Fscaled was obtained by:
Fscaled = Fraw − Vl
Vu − Vl

(2)

.

C. IntegratedStructuredSparseFeatureSelectionUsing
MutualInformation

Feature selection techniques are widely used to improve
model performance and promote generalization in order to
gain a deeper insight into the underlying processes or problem.
This is accomplished by identifying the most important deci-
sion variables, while avoiding overﬁtting a model. Most feature

Authorized licensed use limited to: Michigan State University. Downloaded on November 10,2025 at 21:05:52 UTC from IEEE Xplore.  Restrictions apply. 

HOSSEINI etal.: FNIRS-BASED FEATURE LEARNING AND CLASSIFICATION FRAMEWORK

1257

Fig. 4. The process of feature engineering: pre-process input data, features extraction, post-process the features

Fig. 5. Feature selection and tuning the regularization parameters via N-fold cross-validation in order to introduce the promising features (biomarkers).

selection techniques classify into three categories: embedded
methods, wrapper methods, and ﬁlter methods [24]. Both
embedded and wrapper methods seek to optimize the perfor-
mance of a classiﬁer or model. Thus, the feature selection
performance is highly limited to the embedded classiﬁcation
models. Filter feature selection techniques assess the relevance
of features by measuring their intrinsic properties. Widely used
models include correlation-based feature selection [25], fast
correlation-based feature selection [26], minimum redundancy
maximum relevance (mRMR) [27] and information-theoretic-
based feature selection methods [28].

Sparse modeling-based feature selection methods have
gained attention owed to their well-grounded mathematical
theories and optimization analysis. These feature selection
algorithms employ sparsity-inducing regularization techniques,
such as L1-norm constraint or sparse-inducing penalty terms,
for variable selection. To construct more interpretable models,
structured sparse modeling algorithms that consider feature
structures have recently been proposed and show promis-
ing results in many practical applications including brain

imaging analysis,
informatics, gene expression, medical
etc.
the current structured
[29]–[32]. However, most of
sparse modeling algorithms only consider linear relationships
between response variables and predictor variables (features)
in the analysis and may miss complex nonlinear relation-
ships between features and response variables that may be
present. On the other hand, although some ﬁlter or wrap-
per methods have the capability to capture nonlinear rela-
tionships between features and response variables, feature
structures may not be optimally identiﬁed in the feature selec-
tion procedure. Constructing interpretable learning models
with efﬁcient feature selection remains an open and active
research area in the machine learning community. Zhongxin
et al. [33] proposed a feature selection algorithm based on
mutual information (MI) and least absolute shrinkage and
selection operator (LASSO) using L1 regularization with
application to microarray data produced by gene expres-
sion. In our previous study, we also proposed a MI-based
sparse feature selection model for EEG feature selection
and applied it to epilepsy diagnosis [34]. However, feature

Authorized licensed use limited to: Michigan State University. Downloaded on November 10,2025 at 21:05:52 UTC from IEEE Xplore.  Restrictions apply. 

1258

IEEE TRANSACTIONS ON NEURAL SYSTEMS AND REHABILITATION ENGINEERING, VOL. 26, NO. 6, JUNE 2018

structures were not considered during feature selection in
both [33] and [34].

To consider both linear and nonlinear relationships between
features and response variables, while acknowledging feature
structures in feature selection, we propose a novel feature
selection framework that integrates information theory-based
feature ﬁltering and structured sparse learning models to
effectively capture feature dependencies and identify the most
informative feature subset. There are two differences with
respect to earlier studies [33] and [34]: (1) we did not use
regularization techniques like LASSO as the second rank
ﬁltering; rather, we used sparse-inducing regularization to
reveal the second-level feature-response relationships; (2) we
applied structured feature learning by penalizing the feature
groups. We implemented the proposed information-theory-
based structured sparse learning framework to identify the
optimal feature subset as discriminant neurophysiological bio-
markers of stuttering.

1) MutualInformationforFeatureSelection: MI is an index of
mutual dependency between two random variables that quan-
tiﬁes the amount of information obtained about one random
variable from the other random variable [35]. MI effectively
captures nonlinear dependency among random variables and
can be applied to rank features in feature selection prob-
lems [27]. The fundamental objective of MI-based ﬁltering
methods is to retain the most informative features (i.e., with
higher MI) while removing the redundant or less-relevant
features (i.e., with low MI). The mutual information of two
random variables X and Y, denoted by I (X, Y ), is determined
by the probabilistic density functions p(x), p(y), and p(X, Y ):

I (X; Y ) =

(cid:4)

(cid:4)

y∈Y

x∈X

p(x, y) log

(cid:5)

(cid:6)
,

p(x, y)
p(x) p(y)

(3)

2) StructuredSparseFeatureSelection: A sparse model gen-
erates a sparse (or parsimonious) solution using the smallest
number of variables with non-zero weights among all the
variables in the model. One basic sparse model is LASSO
regression, which employs L1 penalty-based regularization
techniques for feature selection [36]. The LASSO objective
function is formulated as follows:

min {(cid:3)Ax − Y(cid:3) + λ1(cid:3)x(cid:3)1} ,

(4)

where A is the feature matrix, Y is the response variable,
λ1 is a regularization parameter and x is the weight vector
to be estimated. The L1 regularization term produces sparse
solutions such that only a few values in the vector x are non-
zero. The corresponding variables with non-zero weights are
the selected features to predict the response variable Y .

Structured Features (Sparse Group LASSO (SGL))
The basic LASSO model, and many L1 regularized mod-
els, assume that features are independent and overlook the
feature structures. However, in most practical applications,
features contain intrinsic structural information, (e.g., disjoint
groups, overlapping groups, tree-structured groups, and graph
networks) [32]. The feature structures can be incorporated into
models to help identify the most critical features and enhance
model performance.

Information Sparse Feature Selec-

Algorithm 1 Mutual
tion (MISS)
1: Rank all features based on mutual information
2: repeat
3:

k1 = k1 + 1
repeat

Divide sorted features to high-MI and low-MI
S ←high-MI
Remove redundant features from S
until k1 features remain after reduction

4:

5:
6:

7:
8:
9: W ←low-MI
10:
11:

Apply sparsity learning to W
k2 ← number of selected features by SGL or LASSO
Build classiﬁer model with k1 + k2 selected features

12:
13: until classiﬁer performance converges

As outlined in section 2.2,

the features we extracted
thus they can
from the raw fNIRS data are disparate;
be categorized into disjoint groups. The sparse group
LASSO regularization algorithm promotes sparsity at both
the within- and between-group levels and is formulated
as:

(cid:7)

g(cid:4)

(cid:8)

min

(cid:3)Ax − Y(cid:3) + λ1(cid:3)x(cid:3)1 + λ2

A ∈ Rm×n ,

ωg
i

(cid:3)xGi

(cid:3)2

i=1
y ∈ Rm×1,

x ∈ Rn×1,

(5)

xG1

, . . . , xG g

the weight vector x
(cid:9)
groups:

where
is divided by g non-
(cid:10)
, xG2
overlapping
,
and
ωg
i is the weight i for group g. The parameter λ1
the
penalty for sparse feature selection, and the parameter λ2
is the penalty for sparse group selection (i.e. the weights
of some feature groups will be all zeroes). In cases where
feature groups overlap, the sparse overlapping group LASSO
regularization can be used [37].

is

3) Integrated MI-Sparse Feature Selection Framework: The
objective of our approach is to consider structured feature
dependency while keeping the search process computationally
efﬁcient. To accomplish this, we employed the MI-guided
feature selection framework outlined in Algorithm (1). Given
a number of features k, the subset of top k features ranked
by MI is denoted by S, and the subset of the remaining
features is denoted by W . From S, the optimal feature subset is
selected by exploring the k1 high-MI features which includes
the iterative process of removal of highly-correlated features
with 0.96 threshold. From W , the k2 sparse-model selected
low-MI features. The ﬁnal selected features subset is the set
of (k1 + k2) features which are evaluated based on the cross-
validation classiﬁcation performance. Enumeration of k1 starts
from 1 and ascends until reaching the stopping criteria (i.e.,
when the cross-validation accuracy converges and cannot be
further improved). MISS Algorithm (1) can be applied in two
ways: (1) without group structure, which is a combination of
mutual information and LASSO namely (MILASSO), (2) with
group structure, which is a combination of mutual information
and SGL namely (MISGL).

Authorized licensed use limited to: Michigan State University. Downloaded on November 10,2025 at 21:05:52 UTC from IEEE Xplore.  Restrictions apply. 

HOSSEINI etal.: FNIRS-BASED FEATURE LEARNING AND CLASSIFICATION FRAMEWORK

1259

Fig. 6. The process of choosing the most accurate ML classiﬁcation algorithm with N-fold cross-validation and parameter tuning

D. MachineLearningAlgorithmSelection&Evaluation

We applied established ML algorithms [38] (i.e., support
vector machine (SVM), k-nearest neighbor (kNN), decision
tree, ensemble, and linear discriminant) to assess whether
cerebral hemodynamic features could accurately differentiate
the group of children who stutter from controls. An overview
of the steps involved in feature extraction and model evaluation
is provided in Figure (6).

1) SupportVectorMachines: SVM is considered to be a pop-
ular and promising approach among classiﬁcation studies [39].
It has been used in a variety of biomedical applications; for
example, to detect patterns in gene sequences or to classify
patients according to their genetic proﬁles, with EEG signals in
brain-computer interface systems, and to discriminate hemo-
dynamic responses during visuomotor tasks [17], [40]–[42].
In this study we applied Gaussian radial basis function (RBF)
as the kernel which maps input data x to higher dimensional
space.

2) Bayesian Parameter Optimization: Parameters in each
its performance. We applied
classiﬁer signiﬁcantly affect
Bayesian optimization, part of the Statistics and Machine
Learning Toolbox in Matlab, to optimize hyper-parameters of
classiﬁcation algorithm [43]. By applying Bayesian optimiza-
tion algorithm, we want to minimize a scalar objective function
( f (x) = cross-validation classiﬁcation loss) for the classiﬁer
parameters in a bounded domain.

3) N-Fold Cross-Validation: We applied N-fold cross-
validation (N=5) for training and testing. First, we selected
the features and optimized the parameters of the classiﬁcation
algorithm on the training set then applied the tuned model
on the testing set, see Figure (6). Accuracy is deﬁned as
the ratio of correctly classiﬁed test subjects to the total
number of subjects. Sensitivity is the ratio of children in the
stuttering group correctly identiﬁed as stuttering to all of the

children in the stuttering group. Speciﬁcity is the ratio of
children correctly identiﬁed as controls to the total number
of children in the control group. In this study, we used the
average sensitivity and speciﬁcity values to measure binary
classiﬁcation accuracy for each ML model.

III. RESULTS

Classiﬁer performance is reported for experiment (1) based
on the outcome of the N-fold cross-validation procedure on
the test-set, see Table (I). For experiment (2) classiﬁcation
performance was established on a novel test-set of 14 children
who had recovered from stuttering, see Table (IV).

A. Experiment(1):ChoosingtheBestMLAlgorithm

The most accurate ML algorithm on the raw fNIRS data
was the tree classiﬁer with 77.5 % accuracy. The highest
accuracy obtained after feature extraction and application of
feature selection (MILASSO) was SVM (with RBF kernel)
that achieved 87.5 % accuracy, Table (I). The phase of the
fNIRS trial that distinguished the groups of children was the
talk interval and the source was Oxy-Hb. However in some
cases performance using features derived from Deoxy-Hb
measurements reached comparable accuracy as those from
Oxy-Hb.

B. Experiment(1):ComparingFeatureSelection
Algorithms

In Table (II), we compared the performance of the proposed
feature selection algorithm (MISS) with the popular existing
MI-based method like mRMR and linear regularized methods
like LASSO and SGL. MISS approach outperformed mRMR
in feature selection by yielding higher SVM classiﬁcation
performance with the same number of selected features,

Authorized licensed use limited to: Michigan State University. Downloaded on November 10,2025 at 21:05:52 UTC from IEEE Xplore.  Restrictions apply. 

1260

IEEE TRANSACTIONS ON NEURAL SYSTEMS AND REHABILITATION ENGINEERING, VOL. 26, NO. 6, JUNE 2018

TABLE I
COMPARISON AMONG PERFORMANCE OF VARIOUS ML CLASSIFIERS (BEFORE & AFTER) FEATURE
EXTRACTION AND APPLICATION OF FEATURE SELECTION

TABLE II
COMPARISON AMONG PERFORMANCE OF VARIOUS FEATURE SELECTION ALGORITHMS VIA SVM CLASSIFICATION
ACCURACY ON THE SELECTED FEATURES WITH EACH APPROACH

(14 and 11 for measurement source of deoxy-Hb and oxy-Hb),
approximately 7.5 and 27.5 % respectively. MISS approach
outperformed LASSO and SGL in feature selection yielding
higher classiﬁcation accuracy approximately 2.5 to 12.5 %.

C. Experiment(1):SelectedFeatures

From an extended set of features, a subset that provided
the highest classiﬁcation accuracy was identiﬁed by MISGL
and MILASSO in the SVM(RBF) model. This subset of
features, shown in Figure (7), comprises statistical, NAUS,
Hjorth parameters, autocorrelation and bicorrelation features.
Channels that provided the highest discriminative power to
differentiate between children who stutter and controls were
localized to the left hemisphere; speciﬁcally, channels 1, 4,
and 5 over left IFG.

The top 14 features from the entire feature set are listed
in Table (III). These features, (2 based on MI and 12 based
on LASSO), were extracted from the talk-phase with source
Oxy-Hb. We performed 2-tailed t-tests on these features.
p-values ≤ 0.05, conﬁrm a signiﬁcant statistical difference
between children who stutter and controls for a given feature.
1) Feature Selection Optimization: The number of features
selected by MILASSO or MISGL affects the performance
of the classiﬁer; a more sparse selection enhances model
performance, promotes generalization, and facilitates the inter-
pretation of results. During the enumeration process for MI

selection, we learned that with less than 10 MI features (total
features ≤ 15 − 22), the average classiﬁer performance was
approximately 80 %; with 15 to 30 MI features (25−35 ≤ total
features ≤ 40), performance was approximately 75 %; with
more than 30 MI features, (total features ≥ 42 ), the accuracy
decreased to 70 %. The highest accuracy with the least number
of features came from 11 total features with the MILASSO
approach, 2 MI and 9 LASSO and 12 total features with the
MISGL approach, 8 MI and 4 SGL.

2) Biomarkers: The features in Table (III) that showed sig-
niﬁcant differences between children who stutter and controls
are recognized as biomarkers. Box-plots of these features for
the children who stutter and controls are plotted on a common
scale in Figure (8). The discriminative features we detected in
Figure (8) comprised signiﬁcantly lower values of NAUS and
slightly higher values of Hjorth mobility and bicorrelation with
2 sec of delay for children who stutter compared to controls.

D. Experiment(2):StutteringRecoveryAssessmentWith
SelectedFeatures

In this section we report the performance of the classi-
ﬁer on the additional test-set (data from 14 children who
recovered from stuttering), shown in Table (IV). We applied
the best-performing algorithm based on the results from
experiment (1): SVM with tuned parameters sigma = 1 and
penalty = 0.001 on the entire dataset. We documented that

Authorized licensed use limited to: Michigan State University. Downloaded on November 10,2025 at 21:05:52 UTC from IEEE Xplore.  Restrictions apply. 

HOSSEINI etal.: FNIRS-BASED FEATURE LEARNING AND CLASSIFICATION FRAMEWORK

1261

Fig. 7. Statistical summary of the selected feature groups and channels with MILASSO and MISGL in N-fold cross validation. In each fold, there
was 11 to 14 selected features, from different channels and feature-groups. The pie charts illustrate the group that selected features most frequently
came from. The histograms summarize the channel selection with MISGL and MILASSO. For example, from approximately 60 total features selected
from 5 folds, 6 features were selected from channel 1, and 9 features from channel 4 (either based on MI ranking (yellow bar) or LASSO coefﬁcients
(blue bar) which are stacked for each channel).

TABLE III
TOP 14 FEATURES SELECTED WITH MISS ALONG WITH p-VALUE (0.05 THRESHOLD FOR STATISTICALLY SIGNIFICANT t-TEST). WITH TOP
11 FEATURES, 87.5% ACCURACY WAS ACHIEVED IN N-FOLD CROSS-VALIDATION

71.43 %, or approximately 10 out of 14 children who had
recovered from stuttering, classiﬁed into the control group
based on features derived from fNIRS signals derived from
the talk-phase of the experiment. The same degree of stuttering
recovery assessment (SRA) was achieved with both Oxy-Hb
and Deoxy-Hb sources Table (IV).

IV. DISCUSSION

In experiment (1), we applied structured sparse feature
learning models to previously collected speech-evoked fNIRS
data from Walsh et al. [13] to explore whether neurophys-
iological biomarkers could accurately classify hemodynamic
patterns from children who do and do not stutter. Following
feature extraction and feature selection with MISS, the SVM
achieved the highest classiﬁcation accuracy of 87.5 %. With
this model, classiﬁcation performance was improved by 10 %
using feature extraction and sparse MI-based features selec-
tion. This degree of accuracy was reached using features
extracted during the talk interval of the trial from the source,

Fig. 8. Box-plot of top 5 signiﬁcant features from talk-phase and source
Oxy-Hb, ch: channel, (S: stutterer, C: control).

Oxy-Hb (although features extracted from Deoxy-Hb reached
comparable accuracy). A feature set comprising statistics,
NAUS, Hjorth parameters, autocorrelation and bicorrelation
features provided the highest discriminative power. Notably,

Authorized licensed use limited to: Michigan State University. Downloaded on November 10,2025 at 21:05:52 UTC from IEEE Xplore.  Restrictions apply. 

1262

IEEE TRANSACTIONS ON NEURAL SYSTEMS AND REHABILITATION ENGINEERING, VOL. 26, NO. 6, JUNE 2018

TABLE IV
THE BEST SVM PERFORMANCE ON THE ADDITIONAL TEST-SET
(RECOVERED SAMPLES)

nearly all of these features were extracted from channels
localized to the left hemisphere (i.e. channels 1-9). The
selected features may not be signiﬁcant individually as shown
in Table (III), thus they can be ignored or missed in basic
statistical analyses used by many feature selection algorithms.
The MISS approach is valuable to reveal clear discriminative
patterns among features in a higher dimensional space, and to
discover relevant multivariate biomarkers.

Features from channels 1, 4 and 5, which span left IFG,
were identiﬁed as neurophysiological biomarkers that distin-
guished hemodynamic characteristics of children who stutter
from controls. These included signiﬁcantly reduced NAUS
in left IFG channels 4 and 5 and increased Hjorth mobility
parameters, denoting increased variability, in left IFG channels
1 and 4 in children who stutter.

In our earlier study [13], we found signiﬁcantly reduced
Oxy-Hb and increased Deoxy-Hb concentrations during the
talk interval in channels over left IFG in the group of children
who stutter. The left IFG comprising Broca’s area is integral to
speech production and may develop atypically in children who
stutter. Neuroanatomical studies reveal aberrant developmental
trajectories of white and gray matter of left IFG in children
who stutter compared to controls [44], [45]. Moreover, there
is evidence of reduced activation of IFG/Broca’s area during
speech production from fMRI studies with adults who stut-
ter [46], [47]. In our earlier study [13], we hypothesized that
this ﬁnding may represent a shift in blood ﬂow to regions out-
side of our recording area to compensate for functional deﬁcits
in left IFG. An alternative possibility is a disruption in cortical-
subcortical loops resulting in a net inhibition of this region.
This is the ﬁrst study to elucidate group-level differences by
classifying individual children as either stuttering or not stut-
tering using features derived from their speech-evoked brain
hemodynamics. Based on the sensitivity index from the ﬁnal
model, three children who stutter classiﬁed as controls (i.e.,
false negatives). Interestingly, two of these three children were
considered to be mild stutterers when they participated and
have since recovered from stuttering (determined via a follow-
up visit or through parental report). It is tempting to speculate
that the recovery process had already begun for these children
when we recorded their hemodynamic responses during the
initial study. However, longitudinal studies in younger children
(i.e., near the onset of stuttering) are necessary to track the
developmental trajectories of their hemodynamic responses as
they either recover from or persist in stuttering to empirically
assess this assumption.

Finally, we compared the consistency of the best-performing
SVM classiﬁer using N-fold cross-validation from experiment
(1) with results achieved using the SVM classiﬁer on a
novel test-set of data from 14 children who had recovered

from episodes of early childhood stuttering in experiment (2).
We found that the majority of the recovered children, or 71.43
%, classiﬁed as controls, rather than children who stutter.
This suggests that left-hemisphere stuttering biomarkers that
dissociated stuttering children’s speech-evoked hemodynamic
patterns from controls, may indicate chronic stuttering, while
recovery from stuttering in many of these children was asso-
ciated with hemodynamic responses similar to those from
children who never stuttered. Stuttering recovery may thus
be supported, in part, by functional reorganization of regions
such as left IFG that corrects anomalous brain activity pat-
terns. Although this speculation warrants further study and
replication, an fMRI study with adults who recovered from
stuttering identiﬁed the left IFG as a pivotal region associated
with optimal stuttering recovery [48].

A ﬁnal point to consider is that although most of the recov-
ered children had hemodynamic patterns similar to controls,
four of these children classiﬁed into the stuttering group. Given
that stuttering is highly heterogeneous, with multiple factors
implicated in the onset and chronicity of the disorder [2],
it is not surprising to ﬁnd evidence suggesting that recovery
processes may be different for some children. More research
is clearly needed to substantiate the neural reorganization that
accompanies both spontaneous and therapy-assisted recovery
from stuttering.

V. CONCLUSION

In this ﬁnal section, we present several suggestions regard-
ing data preprocessing, feature selection and ML training
and evaluation to guide future investigations in this line of
research.

First, the personalized feature scaling approach facilitated
the discovery of discriminative patterns by removing data
outliers and reducing the variability in each feature. This
was a critical step in our approach to address inherent inter-
individual differences in the physiological signals.

Second, the MISS approach yielded a ﬁnal feature space
that was both parsimonious and interpretable. In particular,
MISGL, that considers feature group structures in sparse fea-
ture learning, and achieved the best classiﬁcation performance
with the least number of selected features. We compared our
result from the MISS approach with commonly used feature
selection techniques in Table (II), and the results proved
that MISS outperformed the methods which solely applied
either MI or regularized linear regression signiﬁcantly. More
importantly, MISS pinpointed speciﬁc left hemisphere chan-
nels that classiﬁed children as stuttering/nonstuttering with
higher accuracy and corroborated ﬁndings from our earlier
experiment [13].

In summary, the proposed MI-based structured sparse fea-
ture learning method demonstrates its effectiveness to dis-
cover the most discriminative features in a high dimensional
feature space with a limited number of training samples,
a common challenge for health care and medical data mining
approaches. Compared to other methods, the proposed MISS
approach offers a promising, interpretable solution to facilitate
data-driven advances in clinical and experimental research
applications.

Authorized licensed use limited to: Michigan State University. Downloaded on November 10,2025 at 21:05:52 UTC from IEEE Xplore.  Restrictions apply. 

HOSSEINI etal.: FNIRS-BASED FEATURE LEARNING AND CLASSIFICATION FRAMEWORK

1263

REFERENCES

[1] O. Bloodstein and N. B. Ratner, A Handbook on Stuttering, 6th ed.

Clifton Park, NY, USA: Cengage Learning, Oct. 2008.

[2] A. Smith and C. Weber, “How stuttering develops: The multifactorial
dynamic pathways theory,” J. Speech, Lang., Hearing Res., vol. 60,
pp. 2483–2505, Apr. 2017.

[3] A. Smith, “Stuttering: A uniﬁed approach to a multifactorial, dynamic
disorder,” in Stuttering Research and Practice: Bridging the Gap. Hove,
U.K.: Psychology Press, 1999.

[4] E. Yairi and N. G. Ambrose, Early Childhood Stuttering for Clinicians

by Clinicians, 1st ed. Austin, TX, USA: PRO-ED, Nov. 2005.

[5] H. Månsson, “Childhood stuttering:

Incidence and development,”

J. Fluency Disorders, vol. 25, no. 1, pp. 47–57, Mar. 2000.

[6] B. Guitar, Stuttering: An Integrated Approach to Its Nature and Treat-

ment, 3rd ed. Baltimore, MD, USA: LWW, Oct. 2006.

[7] E. Blumgart, Y. Tran, and A. Craig, “Social anxiety disorder

in
adults who stutter,” Depression Anxiety, vol. 27, no. 7, pp. 687–692,
Jul. 2010.

[8] J. F. Klein and S. B. Hood, “The impact of stuttering on employment
opportunities and job performance,” J. Fluency Disorders, vol. 29, no. 4,
pp. 255–273, Jan. 2004.

[9] S. O’Brian, M. Jones, A. Packman, R. Menzies, and M. Onslow,
“Stuttering severity and educational attainment,” J. Fluency Disorders,
vol. 36, no. 2, pp. 86–92, Jun. 2011.

[10] L. Iverach and R. M. Rapee, “Social anxiety disorder and stuttering:
Current status and future directions,” J. Fluency Disorders, vol. 40,
pp. 69–82, Jun. 2014.

[11] E. Usler, A. Smith, and C. Weber, “A lag in speech motor coordination
during sentence production is associated with stuttering persistence in
young children,” J. Speech, Lang., Hearing, vol. 60, no. 1, pp. 51–61,
2017.

[12] R. Mohan and C. Weber, “Neural systems mediating processing of sound
units of language distinguish recovery versus persistence in stuttering,”
J. Neurodevelop. Disorders, vol. 7, no. 1, p. 28, 2015.

[13] B. Walsh, F. Tian, J. A. Tourville, M. A. Yücel, T. Kuczek, and
A. J. Bostian, “Hemodynamics of speech production: An fNIRS investi-
gation of children who stutter,” Sci. Rep., vol. 7, Jun. 2017, Art. no. 4034.
[14] A. Villringer and B. Chance, “Non-invasive optical spectroscopy and
imaging of human brain function,” Trends Neurosci., vol. 20, no. 10,
pp. 435–442, Oct. 1997.

[15] F. Homae, “A brain of two halves: Insights into interhemispheric orga-
nization provided by near-infrared spectroscopy,” NeuroImage, vol. 85,
pp. 354–362, Jan. 2014.

[16] W. A. Chaovalitwongse, R. S. Pottenger, S. Wang, Y.-J. Fan, and
L. D. Iasemidis, “Pattern- and network-based classiﬁcation techniques
for multichannel medical data signals to improve brain diagnosis,”
IEEE Trans. Syst., Man, Cybern. A, Syst., Humans, vol. 41, no. 5,
pp. 977–988, Sep. 2011.

[17] S. Wang, Y. Zhang, C. Wu, F. Darvas, and W. Chaovalitwongse, “Online
prediction of driver distraction based on brain activity patterns,” IEEE
Trans. Intell. Transp. Syst., vol. 16, no. 1, pp. 136–150, Feb. 2015.
[18] K. Kam, J. Schaeffer, S. Wang, and H. Park, “A comprehensive feature
and data mining study on musician memory processing using EEG
signals,” in Proc. Int. Conf. Brain Health Inform., 2016, pp. 138–148.
[19] K. M. Puk, K. C. Gandy, S. Wang, and H. Park, “Pattern classiﬁcation
and analysis of memory processing in depression using EEG signals,”
in Proc. Int. Conf. Brain Health Inform., 2016, pp. 124–137.

[20] T. J. Huppert, R. D. Hoge, S. G. Diamond, M. A. Franceschini, and
D. A. Boas, “A temporal comparison of BOLD, ASL, and NIRS
hemodynamic responses to motor stimuli in adult humans,” NeuroImage,
vol. 29, no. 2, pp. 368–382, Jan. 2006.

[21] X. Cui, S. Bray, and A. L. Reiss, “Functional near infrared spectroscopy
(NIRS) signal improvement based on negative correlation between oxy-
genated and deoxygenated hemoglobin dynamics,” NeuroImage, vol. 49,
no. 4, pp. 3039–3046, Feb. 2010.

[22] M. M. Plichta, S. Heinzel, A.-C. Ehlis, P. Pauli, and A. J. Fallgatter,
“Model-based analysis of rapid event-related functional near-infrared
spectroscopy (NIRS) data: A parametric validation study,” NeuroImage,
vol. 35, no. 2, pp. 625–634, Apr. 2007.

[23] S. Wang, J. Gwizdka, and W. A. Chaovalitwongse, “Using wireless EEG
signals to assess memory workload in the n-back task,” IEEE Trans.
Human-Mach. Syst., vol. 46, no. 3, pp. 424–435, Jun. 2016.

[24] Y. Saeys,

I. Inza, and P. Larrañaga, “A review of feature selec-
tion techniques in bioinformatics,” Bioinformatics, vol. 23, no. 19,
pp. 2507–2517, 2007.

[25] M. Hall, “Correlation-based feature selection for machine learning,”
Ph.D. dissertation, Dept. Comput. Sci., Univ. Waikato, Hamilton,
New Zealand, 1999.

[26] L. Yu and H. Liu, “Efﬁcient feature selection via analysis of rele-
vance and redundancy,” J. Mach. Learn. Res., vol. 5, pp. 1205–1224,
Dec. 2004.

[27] H. Peng, C. Ding, and F. Long, “Feature selection based on mutual
information criteria of max-dependency, max-relevance, and min-
redundancy,” IEEE Trans. Pattern Anal. Mach. Intell., vol. 27, no. 8,
pp. 1226–1238, Aug. 2005.

[28] J. R. Vergara and P. A. Estévez, “A review of feature selection methods
based on mutual information,” Neural Comput. Appl., vol. 24, no. 1,
pp. 175–186, Jan. 2014.

[29] F. Liu, S. Wang, J. Rosenberger, J. Su, and H. Liu, “A sparse dictionary
learning framework to discover discriminative source activations in EEG
brain mapping,” in Proc. AAAI, 2017, pp. 1431–1437.

[30] C. Xiao, S. Wang, L. Zheng, X. Zhang, and W. A. Chaovalitwongse,
“A patient-speciﬁc model for predicting tibia soft tissue insertions from
bony outlines using a spatial structure supervised learning framework,”
IEEE Trans. Human-Mach. Syst., vol. 46, no. 5, pp. 638–646, Oct. 2016.
[31] C. Xiao et al., “An integrated feature ranking and selection framework
for ADHD characterization,” Brain Inform., vol. 3, no. 3, pp. 145–155,
Sep. 2016.

[32] J. Gui, Z. Sun, S. Ji, D. Tao, and T. Tan, “Feature selection based on
structured sparsity: A comprehensive study,” IEEE Trans. Neural Netw.
Learn. Syst., vol. 28, no. 7, pp. 1490–1507, Jul. 2017.

[33] W. Zhongxin, S. Gang, Z. Jing, and Z. Jia, “Feature selection algorithm
based on mutual information and Lasso for microarray data,” Open
Biotechnol. J., vol. 10, pp. 278–286, Oct. 2016.

[34] S. Wang, C. Xiao, J. Tsai, W. Chaovalitwongse, and T. J. Grabowski,
“A novel mutual-information-guided sparse feature selection approach
for epilepsy diagnosis using interictal EEG signals,” in Proc. Int. Conf.
Brain Health Inform., 2016, pp. 274–284.

[35] T. M. Cover and J. A. Thomas, Elements of Information Theory (Wiley
Series in Telecommunications and Signal Processing), 2nd ed. Hoboken,
NJ, USA: Wiley, 2001.

[36] R. Tibshirani, “Regression shrinkage and selection via the lasso,” J. Roy.

Statist. Soc. B, Methodol., vol. 58, no. 1, pp. 267–288, 1996.

[37] J. Liu and J. Ye, “Moreau-Yosida regularization for grouped tree
structure learning,” in Proc. Adv. Neural Inf. Process. Syst., 2010,
pp. 1459–1467.

[38] X. Wu et al., “Top 10 algorithms in data mining,” Knowl. Inf. Syst.,

vol. 14, no. 1, pp. 1–37, Jan. 2008.

[39] B. Schölkopf and A. J. Smola, Learning With Kernels: Support Vector
Machines, Regularization, Optimization, and Beyond (Adaptive Com-
putation and Machine Learning). Cambridge, MA, USA: MIT Press,
Dec. 2001.

[40] S. Wang, C.-J. Lin, C. Wu, and W. A. Chaovalitwongse, “Early detection
of numerical typing errors using data mining techniques,” IEEE Trans.
Syst., Man, Cybern. A, Syst., Humans, vol. 41, no. 6, pp. 1199–1212,
Nov. 2011.

[41] M. P. S. Brown et al., “Knowledge-based analysis of microarray gene
expression data by using support vector machines,” Proc. Nat. Acad.
Sci. USA, vol. 97, no. 1, pp. 262–267, 2000.

[42] W. S. Noble, Support Vector Machine Applications in Computational
Biology (Computational Molecular Biology). Cambridge, MA, USA:
MIT Press, 2004, ch. 3.

[43] M. Gelbart, J. Snoek, and R. P. Adams. (Mar. 2014). “Bayesian
[Online]. Available:

constraints.”

unknown

optimization with
https://arxiv.org/abs/1403.5607

[44] D. S. Beal, J. P. Lerch, B. Cameron, R. Henderson, V. L. Gracco, and
L. F. De Nil, “The trajectory of gray matter development in Broca’s area
is abnormal in people who stutter,” Frontiers Human Neurosci., vol. 9,
p. 89, Mar. 2015.

[45] S. Chang, D. C. Zhu, A. L. Choo, and M. Angstadt, “White matter
neuroanatomical differences in young children who stutter,” Brain,
vol. 138, no. 3, pp. 694–711, 2015.

[46] S.-E. Chang, M. K. Kenney, T. M. J. Loucks, and C. L. Ludlow, “Brain
activation abnormalities during speech and non-speech in stuttering
speakers,” NeuroImage, vol. 46, no. 1, pp. 201–212, May 2009.
[47] N. E. Neef, C. Bütfering, A. Anwander, A. D. Friederici, W. Paulus,
and M. Sommer, “Left posterior-dorsal area 44 couples with parietal
areas to promote speech ﬂuency, while right area 44 activity promotes
the stopping of motor responses,” NeuroImage, vol. 142, pp. 628–644,
Nov. 2016.

[48] C. A. Kell et al., “How the brain repairs stuttering,” Brain, vol. 132,

no. 10, pp. 2747–2760, Oct. 2009.

Authorized licensed use limited to: Michigan State University. Downloaded on November 10,2025 at 21:05:52 UTC from IEEE Xplore.  Restrictions apply.
```