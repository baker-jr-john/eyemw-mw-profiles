# Eye Mind Wander Dataset - Complete Data Dictionary

Source: [https://www.eyemindwander.com/key/](https://www.eyemindwander.com/key/)

Each row represents one mind wandering event with corresponding feature values. Events may contain multiple probes measuring task-unrelated thought and boredom. Missing values appear as empty cells due to varying study designs.

## Identifiers

| Variable | Description |
|----------|-------------|
| StudyNum | Study identifier |
| ParticipantNum | Participant identifier |

## Eye-tracker Features

| Variable | Description/Coding |
|----------|-------------------|
| ScreenHeight | Height of screen in pixels |
| ScreenWidth | Width of screen in pixels |
| DeviceType | 1=computer, 2=laptop, 3=phone, 4=VR headset |
| IOS | 1=Windows, 2=Mac, 3=Linux, 4=Other |
| EyeTrackerType | 1=commercial, 2=webcam-based, 3=VR headset |
| RefreshRate | Screen refresh rate |
| SamplingRate | Eye tracker sampling rate |

## Task Features

| Variable | Description/Coding |
|----------|-------------------|
| TaskType | 1=reading, 2=listening, 3=math, 4=video, 5=other |
| PerformanceBinary | 1=yes, 0=no |
| PerformanceDirection | 1=higher better, 0=lower better |
| TaskPerformance | Performance measure during probed task |

## Thought Features - TUT (Task-Unrelated Thought)

| Variable | Description/Coding |
|----------|-------------------|
| ProbeNum | Probe event identifier |
| TUTResponseType | 1=Probe, 2=Self-report, 3=Questionnaire |
| TUTProbeType | 1=yes/no, 2=scale |
| TUTScaleDirection | 1=increasing, 2=decreasing |
| TUTScaleMin | Minimum scale value |
| TUTScaleMax | Maximum scale value |
| TUTResponse | 1=yes, 0=no |
| TUTAggregated | 1=aggregate, 0=not aggregate |

## Thought Features - Intentionality

| Variable | Description/Coding |
|----------|-------------------|
| IntentionalityResponseType | 1=Probe, 2=Self-report, 3=Questionnaire |
| IntentionalityProbeType | 1=yes/no, 2=scale |
| IntentionalityScaleDirection | 1=increasing, 2=decreasing |
| IntentionalityScaleMin | Minimum scale value |
| IntentionalityScaleMax | Maximum scale value |
| IntentionalityResponse | 1=yes, 2=no |
| IntentionalityAggregated | 1=yes, 2=no |

## Thought Features - Awareness

| Variable | Description/Coding |
|----------|-------------------|
| AwarenessResponseType | 1=Probe, 2=Self-Report, 3=Questionnaire |
| AwarenessProbeType | 1=yes/no, 2=scale |
| AwarenessScaleDirection | 1=increasing, 2=decreasing |
| AwarenessScaleMin | Minimum scale value |
| AwarenessScaleMax | Maximum scale value |
| AwarenessResponse | 1=yes, 2=no |
| AwarenessAggregated | 1=yes, 0=no |

## Thought Features - Freely-Moving Thought (FMT)

| Variable | Description/Coding |
|----------|-------------------|
| FMTResponseType | 1=Probe, 2=Self-Report, 3=Questionnaire |
| FMTProbeType | 1=yes/no, 2=scale |
| FMTScaleDirection | 1=increasing, 2=decreasing |
| FMTScaleMin | Minimum scale value |
| FMTScaleMax | Maximum scale value |
| FMTResponse | 1=yes, 0=no |
| FMTAggregated | 1=yes, 0=no |

## Thought Features - Disengagement

| Variable | Description/Coding |
|----------|-------------------|
| DisengagementResponseType | 1=Probe, 2=Self-Report, 3=Questionnaire |
| DisengagementProbeType | 1=yes/no, 2=scale |
| DisengagementScaleDirection | 1=increasing, 2=decreasing |
| DisengagementScaleMin | Minimum scale value |
| DisengagementScaleMax | Maximum scale value |
| DisengagementResponse | 1=yes, 0=no |
| DisengagementAggregated | 1=yes, 0=no |

## Thought Features - Valence

| Variable | Description/Coding |
|----------|-------------------|
| ValenceResponseType | 1=Probe, 2=Self-Report, 3=Questionnaire |
| ValenceProbeType | 1=yes/no, 2=scale |
| ValenceScaleDirection | 1=increasing, 2=decreasing |
| ValenceScaleMin | Minimum scale value |
| ValenceScaleMax | Maximum scale value |
| ValenceResponse | 1=yes, 0=no |
| ValenceAggregated | 1=yes, 0=no |

## Thought Features - Arousal

| Variable | Description/Coding |
|----------|-------------------|
| ArousalResponseType | 1=Probe, 2=Self-Report, 3=Questionnaire |
| ArousalProbeType | 1=yes/no, 2=scale |
| ArousalScaleDirection | 1=increasing, 2=decreasing |
| ArousalScaleMin | Minimum scale value |
| ArousalScaleMax | Maximum scale value |
| ArousalResponse | 1=yes, 0=no |
| ArousalAggregated | 1=yes, 0=no |

## Thought Features - Boredom

| Variable | Description/Coding |
|----------|-------------------|
| BoredomResponseType | 1=Probe, 2=Self-Report, 3=Questionnaire |
| BoredomProbeType | 1=yes/no, 2=scale |
| BoredomScaleDirection | 1=increasing, 2=decreasing |
| BoredomScaleMin | Minimum scale value |
| BoredomScaleMax | Maximum scale value |
| BoredomResponse | 1=yes, 0=no |
| BoredomAggregated | 1=yes, 0=no |

## Gaze Features

| Variable | Description |
|----------|-------------|
| Gazes | Number of gazes |
| Fixations | Number of fixations |
| FixationTime | Time spent fixating |
| UniqueGazes | Number of unique gazes |
| UniqueGazeProportion | Proportion of unique gazes |
| UniqueFixations | Number of unique fixations |
| UniqueFixProportion | Proportion of unique fixations |
| OffscreenGazes | Number of offscreen gazes |
| OffScreenGazeProportion | Proportion of offscreen gazes |
| OffScreenGazeTime | Time spent looking offscreen |
| OffScreenFixations | Number of offscreen fixations |
| OffScreenFixProportion | Proportion of offscreen fixations |
| OffScreenFixTime | Time spent fixating offscreen |
| AOIGazes | Number of gazes in area of interest |
| AOIGazeProportion | Proportion of gazes in AOI |
| AOIGazeTime | Time spent in area of interest |
| AOIFixations | Number of fixations in AOI |
| AOIFixationProportion | Proportion of fixations in AOI |
| AOIFixationTime | Time spent fixating in AOI |

## Demographic Features

| Variable | Description/Coding |
|----------|-------------------|
| Age | Age in years |
| Gender | 1=male, 2=female, 3=nonbinary/other |
| White/Caucasian | 1=yes |
| Hispanic/Latino | 1=yes |
| Black/African American | 1=yes |
| Alaskan/Native American | 1=yes |
| Asian/Pacific Islander | 1=yes |
| Native Hawaiian/Pacific Islander | 1=yes |
| East Asian | 1=yes |
| Southeast Asian | 1=yes |
| South Asian | 1=yes |
| Middle Eastern/North African | 1=yes |
| Medication | 1=yes, 0=no |
| Neurodivergent | 1=yes, 0=no |
| Clinical | 1=clinical, 2=subclinical |
| Glasses | 1=yes, 0=no |

## Study Features

| Variable | Description/Coding |
|----------|-------------------|
| ExpSetting | 1=lab, 2=home, 3=classroom, 4=public setting |
| Alone | 1=yes, 0=no |
| Lighting | 1=well lit, 2=dim lit, 3=no lighting |
