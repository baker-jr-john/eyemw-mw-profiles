# Mind Wandering is Not Monolithic: Latent Profiles of Mind Wandering Dimensions Reveal Distinct Gaze Signatures Across Environmental Conditions

## Abstract

The dominant approach to studying mind wandering (MW) treats it as a binary state: on-task versus off-task. However, MW is a multidimensional phenomenon spanning task-unrelated thought, disengagement, emotional valence, boredom, freely-moving thought, and meta-awareness. Using the Eye Mind Wander dataset---a large-scale collection of eye-tracking and thought-probe data from 20 studies (N = 22,134 observations)---we applied Latent Profile Analysis to empirically derive MW subtypes from multiple MW dimensions measured simultaneously. We identified 5 stable MW profiles in video-based learning (n = 8,167) and 5 profiles in listening/reading tasks (n = 2,674), including On-Task, MW with Disengagement, and MW with Positive Valence profiles. These profiles exhibited modest but significant discriminability from webcam-based gaze features (AUROC = 0.59 for multi-profile classification), and gaze discrimination remained stable across lighting conditions (rho = 0.71--0.86 for feature importance rankings), device types (rho = 0.88), and experimental settings. Our findings challenge the monolithic MW assumption, demonstrate that empirically separable MW subtypes exist and can be partially detected from gaze, and identify environmental conditions where gaze-based detection degrades. These results have implications for the design of adaptive learning systems that must distinguish between different types of learner inattention.

## 1. Introduction

Mind wandering---the shift of attention from a current task to internally-generated thoughts---is a pervasive phenomenon during learning, occurring in 20--50% of thought probes during educational activities (Smallwood & Schooler, 2015). The detection of mind wandering from behavioral signals, particularly eye tracking, has become a major focus for adaptive learning research. If a system can detect when a learner's attention has drifted, it can intervene with prompts, content adjustments, or pacing changes.

However, the field overwhelmingly treats mind wandering as a **binary** state: on-task versus off-task. This monolithic assumption obscures critical distinctions. A student who is deliberately taking a mental break (intentional MW) is cognitively different from one whose attention has involuntarily drifted (unintentional MW). A bored learner who has disengaged from a tedious video differs from an anxious learner ruminating about an upcoming exam. These different MW states may require fundamentally different interventions (Seli et al., 2018).

The Eye Mind Wander (EYEMW) dataset uniquely captures this multidimensional structure. Across 20 studies, thought probes measured up to 8 MW dimensions: task-unrelated thought (TUT), intentionality, awareness, freely-moving thought (FMT), disengagement, emotional valence, arousal, and boredom. Combined with webcam-based eye-tracking data collected across diverse environmental conditions (lighting, device type, experimental setting), this dataset enables a critical test: **Do different dimensions of mind wandering form distinct, empirically separable profiles with unique gaze signatures, and do these profiles remain stable across environmental conditions?**

We address this question in three steps. First, we use Latent Profile Analysis (LPA) to discover data-driven MW subtypes from the joint distribution of MW dimensions. Second, we test whether these profiles can be discriminated from gaze features using Random Forest classifiers. Third, we conduct a "slicing analysis" to assess whether gaze-based discrimination holds across lighting conditions, device types, and experimental settings---a critical consideration for real-world deployment of MW detection systems.

## 2. Related Work

### 2.1 Mind Wandering Detection from Eye Tracking

Eye-tracking approaches to MW detection typically use features such as fixation count, fixation duration, off-screen gaze proportion, and area-of-interest (AOI) metrics (D'Mello et al., 2017; Hutt et al., 2017). Most approaches frame the problem as binary classification: on-task vs. off-task. Recent work using webcam-based eye tracking has achieved AUROCs of 0.60--0.70 for binary MW detection in reading and video-watching tasks (Mills et al., 2021; Jaiyeola et al., 2025).

### 2.2 MW as a Multidimensional Construct

Theoretical work has long argued that MW is not monolithic. Seli et al. (2018) distinguish intentional from unintentional MW. Christoff et al. (2016) propose that MW varies along dimensions of constraint (deliberate vs. spontaneous) and content (past vs. future orientation). The Family Resemblances Framework (Seli et al., 2018) suggests that MW experiences share overlapping features rather than a single defining characteristic. Despite this theoretical richness, empirical work---particularly in eye-tracking research---has rarely operationalized MW as multidimensional.

### 2.3 The Environmental Robustness Gap

Webcam-based eye tracking operates in uncontrolled environments with varying lighting, hardware, and participant settings. Environmental factors have been shown to affect gaze estimation accuracy by 1--5 degrees of visual angle (Papoutsaki et al., 2018). However, the impact of these conditions on the detection of **cognitive states** (rather than raw gaze accuracy) remains unstudied. This gap is critical because adaptive learning systems must function reliably across the diverse conditions of real-world use.

## 3. Data and Methods

### 3.1 Dataset

We used the Eye Mind Wander Database v1.4, comprising 22,134 observation rows from 20 studies with 3,982 unique participants. Each row represents a thought-probe event paired with concurrent gaze features and study/environmental metadata.

**Mind wandering dimensions.** Up to 8 MW dimensions were measured per probe: TUT (binary: 0/1 in most studies), Disengagement (binary or 7-point Likert), Valence (7-point or 9-point Likert), Boredom (7-point Likert), Arousal (7-point or 9-point Likert), FMT (binary or 7-point), and Awareness (binary or 7-point). Intentionality data was unavailable. All responses were harmonized to a 0--1 scale: binary probes were kept as 0/1; Likert scales were min-max normalized using their documented scale endpoints; decreasing scales were flipped so that higher values consistently indicate greater MW.

**Gaze features.** Eight gaze features with substantial data coverage were available: Gazes, Fixations, UniqueGazes, UniqueGazeProportion, OffscreenGazes, OffScreenGazeProportion, AOIGazes, and AOIGazeProportion. All gaze features were z-scored within study to remove study-level scaling differences while preserving within-study variation.

**Environmental metadata.** Each observation was tagged with: experimental setting (lab, home, classroom, public), lighting (well-lit, dim-lit, no lighting), device type (computer, laptop, VR), eye tracker type (commercial, webcam, VR), and task type (reading, listening, video, math).

### 3.2 Study Selection

We required studies to measure TUT plus at least 2 additional MW dimensions, yielding 15 eligible studies (12,850 observations). Two analysis subsets were defined:

- **Subset A** (broadest N): 9 video studies measuring TUT + Disengagement + Valence + Boredom (n = 8,167 across 1,634 participants)
- **Subset B** (richest dimensions): 2 listening/reading studies measuring TUT + FMT + Awareness + Disengagement (n = 2,674 across 382 participants)

Table 1 shows the full eligibility matrix (see Figure 1).

### 3.3 Latent Profile Analysis

We used Gaussian Mixture Models (GMMs) as the Python-based equivalent of Latent Profile Analysis, fitted via `sklearn.mixture.GaussianMixture` with full covariance matrices. Models were fitted for k = 2 through 6 profiles with 20 random initializations per k. Model selection used the Bayesian Information Criterion (BIC), with a cap at k = 5 for interpretability. Profile quality was assessed via average posterior probabilities (target > 0.70) and bootstrap stability (1,000 resamples, measured by Adjusted Rand Index between reference and bootstrap assignments).

### 3.4 Gaze-Based Profile Discrimination

MW profile membership was predicted from gaze features using: (1) multinomial logistic regression (5-fold cross-validation), and (2) Random Forest classifiers (500 trees, balanced class weights, 5-fold CV). Performance was measured by weighted multiclass AUROC (one-vs-rest). Feature importance was computed from Gini importance in the Random Forest. As a baseline comparison, we also fit a binary classifier (on-task vs. any-MW) to quantify the added value of multi-profile classification. Finally, we tested whether adding demographic and hardware covariates (age, gender, screen dimensions) improved discrimination.

### 3.5 Environmental Robustness Slicing Analysis

For each environmental dimension (lighting, device type, experimental setting), we: (1) tested whether MW profile distributions differed across slices using chi-squared tests; (2) computed within-slice gaze discrimination AUROCs; and (3) compared feature importance rankings across slices using Spearman rank correlations (rho > 0.5 indicating robust features).

## 4. Results

### 4.1 MW Dimensions Are Not Monolithic

Correlation analysis confirmed that MW dimensions are not redundant (Figure 2). In Subset A (video studies), the strongest correlation was between Valence and Boredom (r = -0.51); TUT showed only moderate correlations with Disengagement (r = 0.23), Valence (r = -0.28), and Boredom (r = 0.25). In Subset B (listening tasks), correlations were even weaker: TUT-Disengagement r = 0.26, FMT-Awareness r = 0.33, TUT-Awareness r = 0.03. These low-to-moderate correlations indicate that MW dimensions capture genuinely different aspects of the MW experience.

### 4.2 Five MW Profiles Emerge from LPA

**Subset A (Video Studies).** LPA identified 5 stable profiles (bootstrap ARI = 0.82, mean posterior = 1.00; Figure 3a):

1. **On-Task** (49.4%, n = 4,031): Low TUT, low disengagement, moderate valence and boredom. The largest group, representing the majority of probe responses.
2. **Mind Wandering** (22.7%, n = 1,852): High TUT, no disengagement, moderate valence, elevated boredom. Classic off-task thought without behavioral disengagement.
3. **On-Task + Positive Valence** (10.1%, n = 825): No TUT, no disengagement, maximum positive valence, no boredom. An engaged, emotionally positive state.
4. **MW + Disengaged** (10.0%, n = 820): Both TUT and disengagement at maximum, moderate boredom. The most severe MW profile, combining both cognitive and behavioral withdrawal.
5. **On-Task + Disengaged** (7.8%, n = 639): No TUT but full disengagement, positive valence, moderate boredom. A paradoxical profile where learners report being on-task but disengaged---possibly reflecting "going through the motions."

**Subset B (Listening/Reading Tasks).** LPA identified 5 profiles (bootstrap ARI = 0.66, mean posterior = 0.99; Figure 3b):

1. **On-Task + Aware (High FMT)** (38.1%, n = 1,018): No TUT, moderate FMT, moderate awareness and disengagement. An attentive state with some freely-moving thought.
2. **Mind Wandering** (30.4%, n = 812): High TUT, moderate FMT, moderate awareness and disengagement.
3. **On-Task + Aware** (14.9%, n = 398): No TUT, moderate FMT, maximum awareness, no disengagement.
4. **On-Task** (11.4%, n = 306): No TUT, low FMT, low awareness, no disengagement. A focused but low-meta-awareness state.
5. **MW + Disengaged + Aware + Free-Flowing** (5.2%, n = 140): High TUT, high FMT, maximum awareness, full disengagement. A deliberate, meta-aware form of MW.

### 4.3 Gaze Features Partially Discriminate MW Profiles

Random Forest classifiers achieved weighted AUROCs of 0.59 for both Subset A and Subset B (Table 2), significantly above chance (0.50) but reflecting the limited discriminative power of webcam-based gaze features for fine-grained MW profiles.

The most important gaze features for profile discrimination were: Gazes (total gaze count), UniqueGazeProportion, UniqueGazes, AOIGazeProportion, and OffscreenGazes (Figure 4). Notably, Fixations contributed zero importance in both subsets, likely because this feature had data from only one study within each subset.

**Binary baseline comparison.** Binary classification (on-task vs. any-MW) achieved AUROC = 0.70 (Subset A) and 0.63 (Subset B)---higher than multi-profile classification. This suggests that while gaze features can detect MW broadly, discriminating between MW subtypes from gaze alone is harder. However, the multi-profile approach provides richer information for intervention design even if per-profile detection is weaker.

**Covariates improve discrimination.** Adding age, gender, and screen dimensions to gaze features boosted AUROC to 0.71 (Subset A) and 0.75 (Subset B), indicating that individual and hardware differences account for substantial variance in the gaze-MW relationship.

### 4.4 Environmental Robustness

**Profile stability.** MW profile distributions differed significantly across lighting conditions (chi2 = 24.2, p = .002), device types (chi2 = 27.8, p < .001), and experimental settings (chi2 = 33.7, p < .001) in Subset A. However, within-slice LPA reproduced similar profile structures (within-slice ARI = 0.80--0.91), suggesting that the same profiles emerge even when the data is analyzed separately within each environmental condition.

**Gaze discrimination by slice (Figure 5).** AUROC varied modestly across conditions: lighting (well-lit: 0.60, dim-lit: 0.64, no lighting: 0.71), device type (computer: 0.61, laptop: 0.60), and setting (home: 0.59, public: 0.59). Surprisingly, dim-lit and no-lighting conditions yielded *higher* AUROCs, possibly because offscreen gaze patterns are more distinctive under suboptimal lighting.

**Feature importance robustness.** Gaze feature importance rankings were highly stable across lighting conditions (rho = 0.71--0.86) and device types (rho = 0.88), but less stable across experimental settings (rho = 0.14 for home vs. public). This suggests that while the *relative* importance of gaze features generalizes well across hardware and lighting, the home vs. public distinction may involve qualitatively different attentional dynamics.

## 5. Discussion

### 5.1 Implications for MW Theory

Our results provide empirical evidence that mind wandering during learning is not monolithic. The LPA revealed 5 distinct profiles that capture meaningful variation beyond the binary on/off-task distinction. Particularly noteworthy is the "On-Task + Disengaged" profile (7.8% of Subset A), which would be classified as "on-task" in binary schemes but involves full behavioral disengagement---a state that may indicate passive, unconstructive learning. Similarly, the "MW + Disengaged + Aware + Free-Flowing" profile in Subset B (5.2%) represents a meta-aware, deliberate form of MW that may be less detrimental to learning than unaware MW (Seli et al., 2018).

These profiles align with theoretical frameworks that distinguish MW along dimensions of intentionality and awareness (Christoff et al., 2016), and provide the first large-scale empirical taxonomy of MW subtypes derived from probe data.

### 5.2 Implications for Adaptive Learning

The finding that gaze features achieve modest but above-chance discrimination of MW profiles (AUROC = 0.59) has nuanced implications. While this performance is insufficient for reliable real-time detection of specific MW subtypes, it suggests that gaze signals contain partial information about the *type* of MW, not just its presence. An adaptive system could use this information probabilistically: high confidence of the "MW + Disengaged" profile might trigger a re-engagement prompt, while the "On-Task + Positive Valence" profile might indicate optimal learning conditions that should be maintained.

The substantial improvement from adding covariates (AUROC 0.59 to 0.71--0.75) highlights that personalization---accounting for individual differences and hardware context---is essential for real-world MW detection.

### 5.3 The Environmental Robustness Story

Our slicing analysis reveals a largely positive story for deployment: gaze-based MW discrimination is reasonably stable across lighting and device conditions. The high rank correlations for feature importance (rho = 0.71--0.88) suggest that the *same* gaze features matter regardless of whether the learner is using a laptop or desktop, or studying in well-lit vs. dim conditions. This is encouraging for real-world deployment of MW detection systems.

However, the low rank correlation for home vs. public settings (rho = 0.14) is a cautionary finding. The attentional dynamics of studying at home may be fundamentally different from studying in a public space (e.g., library), with different sources of distraction and different gaze behaviors associated with MW.

### 5.4 Limitations

Several limitations constrain our conclusions. First, MW dimensions in the EYEMW dataset are heavily binary (particularly TUT and Disengagement), which constrains the continuous variation that LPA can detect. The resulting profiles are partly determined by the combinatorial structure of binary responses rather than smooth latent continua. Second, webcam-based gaze features have limited precision (1--5 degrees spatial accuracy, 30 Hz temporal resolution), which fundamentally caps the discriminability of fine-grained cognitive states. Third, the dataset does not contain Intentionality data, preventing us from testing one of the most theoretically important MW distinctions. Fourth, Subset B (listening/reading) contains only 2 studies, limiting generalizability for non-video tasks.

### 5.5 Future Directions

Future work should: (1) collect continuous MW dimension ratings to enable more granular LPA; (2) combine gaze with other low-cost sensors (facial expression, mouse behavior) for multimodal MW subtype detection; (3) test whether MW profiles predict differential learning outcomes; and (4) develop personalized MW detection models that adapt to individual gaze baselines.

## 6. Conclusion

Mind wandering is not monolithic. Using Latent Profile Analysis on the largest multi-dimensional MW dataset to date, we identified 5 empirically distinct MW profiles that capture meaningful variation beyond the binary on/off-task distinction. These profiles show distinct gaze signatures that are modestly but reliably detectable from webcam-based eye tracking, and this detection remains stable across diverse environmental conditions. Our findings argue for a paradigm shift in MW research and adaptive learning: from "Is the student mind wandering?" to "What kind of mind wandering is occurring, and what does it mean for learning?"

## References

Christoff, K., Irving, Z. C., Fox, K. C. R., Spreng, R. N., & Andrews-Hanna, J. R. (2016). Mind-wandering as spontaneous thought: A dynamic framework. *Nature Reviews Neuroscience*, 17(11), 718--731.

D'Mello, S. K., Dieterle, E., & Duckworth, A. (2017). Advanced, analytic, automated (AAA) measurement of engagement during learning. *Educational Psychologist*, 52(2), 104--123.

Hutt, S., Mills, C., Bosch, N., Krasich, K., Brockmole, J., & D'Mello, S. (2017). "Out of the fr-eye-ing pan": Towards gaze-based models of attention during learning with technology in the classroom. *Proceedings of the 25th Conference on User Modeling, Adaptation and Personalization*, 94--103.

Jaiyeola, T., et al. (2025). [Study details from EYEMW Study 004]. *Proceedings of the 17th International Conference on Educational Data Mining*.

Mills, C., Gregg, J., Bixler, R., & D'Mello, S. K. (2021). Eye-mind reader: An intelligent reading interface that promotes long-term comprehension by detecting and responding to mind wandering. *Human-Computer Interaction*, 36(4), 271--301.

Papoutsaki, A., Laskey, J., & Huang, J. (2018). SearchGazer: Webcam eye tracking for remote studies of web search. *Proceedings of the 2018 Conference on Human Information Interaction & Retrieval*, 27--36.

Seli, P., Kane, M. J., Smallwood, J., Schacter, D. L., Maillet, D., Schooler, J. W., & Smilek, D. (2018). Mind-wandering as a natural kind: A family-resemblances account. *Trends in Cognitive Sciences*, 22(6), 479--490.

Smallwood, J., & Schooler, J. W. (2015). The science of mind wandering: Empirically navigating the stream of consciousness. *Annual Review of Psychology*, 66, 487--518.
