# Literature research

## Using "CBR fault detection" as search words in ieeeXplore

### Adaptive Maintenance Knowledge Bases for Field Service
url: https://ieeexplore.ieee.org/document/4161647/
- Maintenance Knowledge Base(MKB), an information system capable of offering solutions to diagnostic problems at a level comparable to experts
- The paper presents a solution that "seamlessly combines model-based reasoning (MBR) and CBR for adaptive knowledge base creation, maintenance and update through multi-signal flow graph modeling.
- The solution combines the prior knowledge with the posterior knowledge using a Bayesian framework.
- Reduces the upfront cost in creating the diagnostic model, but also the barrier of entry for system fault detection and diagnosis(FDD)

### A case-based reasoning approach to the management of faults in communications networks
url: https://ieeexplore.ieee.org/document/366653/
- Old, from 1993
- Most systems for FDD are rule-based reasoning(RBR), these systems have no learning to them
- The paper enhances a standard fault management system with a CBR component.
- It reviews retrieving, adapting and retaining, and CRITTER a CBR trouble ticketing system for managing and resolving network faults.

### Using the Case-Based Ranking Methodology for Test Case Prioritization
url: https://ieeexplore.ieee.org/document/4021329/
- Prioritizing test cses to optimize the achievement of the testing goal, test the cases most likely to be faulty first
- Case-based ranking (CBR), elicits just relative priority information from the user, comparisons
- User inputs(cases or just the queries?) are integrated with multiple prioritization indexes.
- Results on a case study indicate that the CBR overcomes previous approaches, and can get close to the optimal solution

### Multiway Principal Component Analysis and Case Base Reasoning methodology for abnormal situation detection in a nutrient removing SBR
url: https://ieeexplore.ieee.org/document/7068413/
- Multiway Principal Component Analysis(MPCA) and CBR are applied in a biological nutrient removal process(removal of unwanted/over represented nutrients from water).
- CBR was used to find similar solutions under normal/abnormal situations: low ORP, high pH.
- Several proof are made in the paper

### Expert supervision based on cases
url: https://ieeexplore.ieee.org/document/996399/
- Take advantage of data stored in SCADA systems(used in processing plants/factories)
- Using a CBR system to perform export supervisory tasks in a dynamic system
- Focuses on proposing a general case definition suitable for supervisory tasks
- Is tested on a real drier chamber  

### Case-Based Reasoning in the Plenary Diagnostic Environment
url: https://ieeexplore.ieee.org/document/4062447/
- CBR on multi-level(Plenary) diagnostic environment: the fault can be on different levels, in manifacturing it can be on a car, the facility or the personel

### Fault diagnosis method for complex equipment using CBR
url: https://ieeexplore.ieee.org/document/6852276/
- FDD on complex equipment to increase reliability and decrease possible loss
- Uses CBR to infer and classify various failures
- Case retrieval based on the improved grey relational analysis
- CBR based FDD-system applied to diesel engine

### Fault detection using topological case based modelimg and its application to chiller performance deterioration
url: https://ieeexplore.ieee.org/document/352030/
- Topological Case Based Modeling (TCBM) as FDD
- Continuous input/output relation, possible to describe nonlinear systems
- CBR inferes from stored cases and these relation
- Example of TCBM to detect deterioration for a chiller system

### Focusing fault localization in model-based diagnosis with case-based reasoning
url: https://ieeexplore.ieee.org/document/7068686/
- Consistency-based diagnosis automatically provides fault detection and localization capabilities, using just models for correct behavior, but lacks discrimination power.
- Studies Consistency-based diagnosis system together with CBR systems
- CBR system provides accurate indication of most probable fault in the early process of the Consistency-based diagnosis system


### Case-base reasoning in vehicle fault diagnostics
url: https://ieeexplore.ieee.org/document/1223990/
- CBR as FDD in vehicles
- Distributed diagnostic agent system (DDAS) as FDD using signal analysis and ML
- CBR used to find root faults in vehicles, based the DDAS. Two CBR methods used, one uses direct sensor data, another the signal segment features
- Experiments conducted on real vehicles

### A case-based reasoning approach for fault detection state in bridges assessment
url: https://ieeexplore.ieee.org/document/4588730/
- CBR as FDD of reinforced concrete bridges on polluted areas.
- Earlier FDD uses fuzzy models and neural networks, this uses CBR, might have a comparison
- Uses jCOLIBRI

### A Case-based Reasoning with Feature Weights Derived by BP Network
url: https://ieeexplore.ieee.org/document/4426957/
- Investigates the performance of a hybrid case-based reasoning method, integrading a multi-layer BP NN with CBR algorithms for dericatives feature weights
- The BP NN obtains attribute weights, CBR classifies
- Studies different parameters to test performance
- Got better results using the hybrid method then CBR alone

### Incremental dictionary learning for fault detection with applications to oil pipeline leakage detection
url: https://ieeexplore.ieee.org/document/6047960/
- Explains and references the negative pressure wave(NPW) method, when a leak happens the preassure drops and this can be used to identify where the leak happend.
- During testing the purposed method had 2 correct detectoins and no false alarms.

### Pipeline leakage monitor system based on Virtual Instrument
url: https://ieeexplore.ieee.org/document/5246155/
- More details on NPW method on leakage detection

### A combined kalman filter — Discrete wavelet transform method for leakage detection of crude oil pipelines
url: https://ieeexplore.ieee.org/document/5274381/
- using kalman filters


#### New lit res:

### Using Correlation between Data from Multiple Monitoring Sensors to Detect Bursts in Water Distribution Systems
url: https://ascelibrary.org/doi/pdf/10.1061/%28ASCE%29WR.1943-5452.0000870
citations: 

## Context:
- Current and good root for further research

## Goal/Motivation:
- Data-driven burst detection methods uses a prediction and a classification stage. This paper proposes a clustering method that only uses data over a smaller time window(i.e. 12 hours).

## Related work:
[0]: Rodrigues and Laio(2014), similarity measures local density and distance to vectors with higher densities
[1]: Colombo et al. 2009, summary of transient-based applications
[2]: Romano et al. 2014a, ANN approach
[3]: Sanz et al. 2016, Wu et al. 2010, model-based approach

## Methodology:
- Data-driven method

## Summary:
- Represents flow and preassure data from each inlet/outlet to the DMA(district metering area) as vectors, normalized by the median of the current detection window. One vector per timestep.
- Then checks correlation between each vector using Cosine Distance, and two quantities defined by [0].
- "An abnormal flow vector is defined characterized by a relatively large distance and the lowest local density." Distence is quantified by a significance factor alpha.
- The initial history matrix(matrix of data-vectors) was "cleaned" by removing outlier vectors with a significance factor alpha_1. Alpha_2 was used for future outlying vectors
- Has a table of reasons for changes in vectors and corresponding change conditions
- Resulted in 0.40% False Positive Rate(FPR) and 71.43% true positive rate(TPR)
- Method removes weather impact on burst detection
- Method is good on small data sets, vs bigger data sets required by other data-driven methods
- Parameters alpha_1, alpha_2 and size of the detection window decieds the results(FPR/TPR) of the method.
- Has two undetectable outliers

## Evaluation/Master ideas:
- Good and well referenced introduction to burst dection, use to find state of the art and related work, refrences other data-driven methods aswell
- Other similarity measures
- Expand and classify individual reasons for changes in vectors, can also be used for explanation
- My data is over more DMA's(hopefully)
- Detect the undetected outliers
- Paper removes weather impact, maybe it should be added?
- Find better standard answer then zero on faulty/missing sensor data
- Does not use burst location(combo wombo?)
- "The data of several pressure sensors within a DMA also shows strong temporal varying correlation", no backup of this other then a graph, also does not test independent DMA's
- Checking transferability in both a bigger system and also between independent DMA's
- Instead of this method, use a RNN with the same windows to solve the problem, should there be a reason behind introducting a new approach?



### Distance-Based Burst Detection Using Multiple Pressure Sensors in District Metering Areas
url: https://ascelibrary.org/doi/pdf/10.1061/%28ASCE%29WR.1943-5452.0001001

## Context:
- Uses a data-driven approach to point out the closest pressure sensor to the burst

## Goal/Motivation:
1. Utilize correlation between pressure sensors
2. Detect burst using the correlation
3. Provide approximate location information of the burst(identify the closest pressure sensor)

## Related work:
[0]: Wu et al (2018), using cosine distance to utilize the correlation between pressure measurements
[1]: Knorr and Ng (1998), distance-based(DB) algorithm used for detecting bursts
[2]: Pudgar and Ligett (1992), abnormality degree(AD) to identify the closest pressure sensor

## Methodology:
- Data-driven, pressure data in 5 min intervals
- Implements cosine distance and AD into the DB-algorithm

## Summary:
- Comes with a new data-driven method focusing on several sensors instead of a single one, and adds some localization information
- Burst: event causing flow rate increase of >40% of average DMA inflow
- Classifies 3 feature changes:
    1. Data from inlet pressure sensor change only slightly
    2. "The temporal varying correlation between the inlet pressure sensor and other sensors disappears during bursts"
    3. The closest pressure sensor has the most significant pressure drop
- Data is represented as (n-1) data streams, where n is the number of pressure sensors, each stream is a 2-d vector containg the inlet pressure messure and the i'th sensor value
- The DB-algorithm is used on the vectors within a sliding window in each datastream. Vectors with few neighbors are prone to be outliners, distance measured using cosine distance
    - DB can be optimized by 3 variables, p, q and l(size of the sliding window)
- Abnormality degree is introduced to be 1-(k/N) if k < N or 0 otherwise. k is the number of neighbors to the vector. => if a vector has few or no neighbours it is an outlier
- One time slice is indentified as an instance, if any outlier is identified in an instance, it is an abnormal instance
- Testing shows that p and q are the variables that impact the FPR and TPR, while l has no effect when it is choosen sufficiently large(such that the window cover several days)
- With p=0.19 and q=0.47 => FPR = 1.6%, but the TPR < 100%. Here all events where identified, but not all 50 instances that corresponded with the events.
- 270 instances where classified as FP, 62 because of missing data, the system identifies 2 or more concecutive abnormal instances as reason for alarm, resulting in 15 false alarms.
    - 3 related to unknown events that caused fluctuating flow and pressure
    - 12 where water usaged during demanding hours, i.e. customer demand caused abrupt increase in water usage.


## Evaluation/Master ideas:
- Find a better solution for missing data then setting sensor to 0
    - dynamic? If no data, just ignore the sensor?
- Paper suggests:
    - Optimizing p and q, suggests genetic algorithm
    - Identifying minimum detectavle burst size
    - Reduce FP connected with customer demand(seen as normal behavior)
    - More precise burst location using the network topology

### Guidelines for Snowballing in Systematic Literature Studies and a Replication in Software Engineering
url: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.709.9164&rep=rep1&type=pdf

## Context:
- Get a thourogh understanding of snowballing and how to present how it was done, and how to actually do it

## Goal/Motivation:
- Literature review using a snowballing approach
1. "Formulate a systematic snowballing procedure for systematic lterature studies in software engineering"
2. "Illusrate and evaluate the snowballing procedure by replicating a published systematic lit. review"

## Related work:
[0] Skoglund and Runeson[16], reference-based research focused on reducing inital papers, not satisfactory results
[1] MacDonell et al. [11], stavle snowballing procedure

## Methodology:
- Replicates a lit. res. using snowballing and compares the results of two lit. res. 

## Summary:
- Needs a systematic approach to build individual knowledge and combine knowledge from diff. sources
- Backward snowballing, evaluate the refrence list
- Forward snowballing, evaluate where the paper is cited
- Process:
    1. Identify start set and include them:
        - Only papers that are going to be used(not candidates for the start set)
        - Avoid bias that favors authors or publishers
        - Independent papers
        - Big enough start set
        - Diverse
        - Also look for relevant papers using synonyms
    2. Iterate:
        - For each included paper do one backward snowball phase
        - For each included paper do one forward snowball phase
        - Tips: View in which context the paper is cited/refrenced
        - Before a paper is included read abstract -> skim paper
        - NB: Never go to the next paper before you are finished with the current
        - Keep the papers iteration specific
        - Contact authors, look at journals/confrences for more papers
- Use google scholar to avoid pulisher bias
- Efficiency tracked using #candidates/#included
- 10 lessons learned:
    - Easy excludsion of papers
    - Avoid clusters and biases
    - Difficulty of exclusion vs evaluating more
    - Inlcude/Exclude may be transitive through context
    - Frequency of inclusion of papers should be tracks per iter
    - If there are no drop in frequency, it indicates a undiscovered cluster
    - Citation matrix are usefull, matrix where citation/refrencing of papers are tracked, also if it was possible to cite/refer based on the timeline


## Evaluation/Master ideas:
- Same people did the two lit. res. that are compared

### LeakDB: A benchmark dataset for leakage diagnosis in waterdistribution networks 
url: https://ojs.library.queensu.ca/index.php/wdsa-ccw/article/view/12315/7911

## Why to read:
- Benchmarking and datasets


### Risk-based sensor placement methods for burst/leak detection in water distribution systems 
url: https://iwaponline.com/ws/article/17/6/1663/38191/Risk-based-sensor-placement-methods-for-burst-leak?searchresult=1

## Why to read:
- The optimal placement of sensors for burst/leak detection in water distribution systems is usually formulated as an optimisation problem. In this study three different risk-based functions are used to drive optimal location of a given number of sensors in a water distribution network. A simple function based on likelihood of leak non-detection is compared with two other risk-based functions, where impact and exposure are combined with the leak detection likelihood

### Short-term water demand forecasting using machine-learning
url: https://iwaponline.com/jh/article-abstract/20/6/1343/63662/Short-term-water-demand-forecasting-using-machine?redirectedFrom=fulltext

## Why to read:
- Might shed some light on how to apply ML on water data


### Artificial neural networks: applications in the drinking water sector
url: https://watermark.silverchair.com/ws018061869.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAgYwggICBgkqhkiG9w0BBwagggHzMIIB7wIBADCCAegGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMPX_JlvQXYjg55dvAAgEQgIIBue-9EWIDM2O6mlD1Ls9GOOFsbUGpx7c86BkgDWAIz6AqO0DZDlEYR7VbnjZNhRgsDVdIoBSlval-J8bazR2kT677DI7ACQG0a75NnOSG0T6fr3Z64RYPC_mRcqyZa_YPCnBrxkqVELqbhuOpjYhudh3SI_MiL0DIOIQtaT-K8VHiLxv8gnJxo__ixb_aGwex9j_Q8yN3ytP75PIXX5980LE5hi6Cl8q3pnKpxDprZJ9w0qrFVV-yfKPYafMUD-SbyPHSlZJZHl5EOdc3xGElb2XhT8Z3yXgM6re0V2w10Uh67mXDNKicQleaXikO6w0mM6qPaKEqHFIOEkZmIvkZFwubj-e1VcsAUchi_lhvudBK_hVtz-fuD469-ar4SsHNCJUPMkbEtMDVX4RNT8CJ_jSybAt5-rzMUEIZBQb4X8RcN0DxPqQTpB-K3tnTg_EiEDRdidJp-uJqhTg2XBscaCxkf7jM8zDV4Fg1Xsm7lvOzl3JcFIOfXaS32C1SyNOW-Xn_Aq2WY-ItUnMpgSofAc5_BZzO5IQ4hhdXy1wOD8hioXcmYtGzZGD4LJ1fBG5fl2KIavjJJ0pZrw


#### Snowballing

### Multivariate Principal Component Analysis and Case-Based Reasoning for monitoring, fault detection and diagnosis in a WWTP(WasteWaterTreatmentPlant)
url: https://iwaponline.com/wst/article/64/8/1661/31553/Multivariate-Principal-Component-Analysis-and-Case?searchresult=1   


### Multi-Leak Detection with Wavelet Analysis in Water Distribution Networks
url: https://ieeexplore.ieee.org/document/6265794

## Context:
- Leak detection and localization using sensitivity matrix and wavelet analysis, joined with a voting system to identify multiple leaks.

## Goal/Motivation:
1. Multiple leak detection and localizations

## Related work:
[0] Ragot: https://ieeexplore.ieee.org/document/6265794, detectes faults in measurements using fuzzy analysis, mentions 25% loss of purified water, but reference goes nowhere.
[1] Mashford: https://ieeexplore.ieee.org/document/5319304, location and size of leaks using SVMs, good list of practicle detection methods
[2] Covas: https://www.civil.ist.utl.pt/~hr/BHR01-03-28_DC-HR_.pdf, leakage detection and location using transitory inverse analysis
[3] Perez: https://www.sciencedirect.com/science/article/pii/S0967066111001201, Model-based leakage location
[4] Casillas: https://ieeexplore.ieee.org/abstract/document/6669568, Model-based leakage detection

## Methodology:
- Model-based
- Uses sensitivity matrices and wavelet analysis to generate a comparision matrix, that is then used with a voting system to localize the leaks

## Summary:
- Focuses on multiple leak detection, instead of the standard single leak detection, is done using 14 steps:
    1. Obtain sensitivity matrices by subracting the pressure matrix with a leak of magnitude l at node j from the pressure matrix without any leaks
    2. Perform the Wavelet Analysis, calculating the wavelet coefficients Cs and Csa from Sm and Sa
    3. Binarize the matrices Cs, Csa and Rab to get Csb, Csab, Rab which has seperated the imaginary and real parts and binirized by checking if the values are greater then 0.
    4. Obtain the comparision Matrix(n,m) =  sum(CSB^(Rk)_(n,m) xor Cbs) and more
- Compared with Angle between vectors; WA is better with measurement noise(noise in pressure) while angle between vectors is better with flow noise.
- But all methods are bad with noise, best results is 93%, but then gets only closest 2. degree neighbour.


## Evaluation/Master ideas:
- Also look at multiple leak detection
- get better detection with added noise

### Model-based leak detection and location in water distribution networks considering an extended-horizon analysis of pressure sensitivities
url: https://iwaponline.com/jh/article/16/3/649/3082/Model-based-leak-detection-and-location-in-water?searchresult=1

## Context:
- 5 different model-based leak detection methods are presented and compared
- Also very good overview of the state of the art

## Goal/Motivation:
- New model-based method compared with 5 others

## Related work:

## Methodology:
- Model-based leak detection method using sensitivity matrices and residues
- Binarized sensitivity method:
    1. Binirize sensitivity matrix using a threshold
    2. Binirize residue based on threshold
    3. Compare the residue vector with each coloumn of the sens matrix, if equal this indicate a leak
    4. The comparisons are stored in a matrix, and a leak indication vector is created by summing the rows of the matrix
    5. The leak node is the index og the biggest component of the leak vector
    6. The leak magnitude can be estimated by finding the S coloumn that lies closest to residual vector
- Angle between bectors method
    1. Find the cosine distance between each coloumn of S and the residue vector
    2. Calculate the mean angle in the time window
    3. Candidate leak is the node with the least mean angle
    4. location is done as step 6
- Correlation method
    1. calculate the correlation between residue and coloums of S (formula in paper)
    2. calculate the mean correlation
    3. Leak is the smallest mean correlation
    4. location is done as step 6
- Euclidean distance method, only works well when the leak has the same magnitude as the one used to compute the sensitivity matrix
    1. calculate the euclidean distance between columns of S and the residue vector
    2. calculate the mean distance
    3. leak node is the one with the smallest mean distance
    4. locatrion as before 
- Least square optimization method, gives an indication of the leak size aswell as the leak node
    1. Oposite of the other solutions, it calcualtes the most appropriate leak size first by optimizing a least square equation found in the paper
    2. Then the leak node is the one with the mininal index

## Summary:
- Bin. method is weak because the threshold is hard to set
- For Euclidean dist the problem is noise and demand patterns
- Cosine dist and least square opt is the best
- Localization:
    - Angle method: Detects 80%, 88% of the leak nodes within 2m, the max is on 700m, mean is 100m, reducing the number of sensors has litle effect
    - Optimization method: 81% and 89% detection with 2m, not as good as angle methods, but gives an approx of the leak size, with is nice
    - Correlation, not as accurate as the others.

### Leakage fault detection in district metered areas of water distribution systems
url: https://iwaponline.com/jh/article/14/4/992/3203/Leakage-fault-detection-in-district-metered-areas

## Context:
- Analyse the inflow to a DMA and learn the weekly periodic DMA inflow dynamic, i.e. a model based approach
- Currently uses Night flow analysis(NFA), this is prone to not detect slowly increasing leaks(would be viewed as demand)
- Problem with model based: may not have well calibrated models, as well as representative consumer demand models
- Transiant analysis requires high freq. sampling

## Goal/Motivation:
- Model based

## Related work:
- Savic et al. (2009) - disadvantages with transient analysis

## Methodology:
- Presented method:
    - Update coefs of fourier series to reprisent demand changes. The offset term is used to id leaks. Found to be good at detecting small leaks
- Split the signal into 2; The longterm signal(yearly, ignored in this paper) and the shortterm(weekly)
- Create a approximation of the weekly demand pattern represented as a fourier series
- The initial approximation is calculated using historical data
- Then "learns" using a learning rule, eq 6 in paper, containing a diagonal learning matrix G, 0 < Gi < 2, has same problems as regular learning rule( too small => slow learning, too big => may change to drastic)
- The leak detection is done using CUSUM on the mean of the estimated weekly demand, and there is a leak if the CUSUM is greater than a treshold
- The treshold depends on te DMA inflow variance
- Leak magnitude is calculated as the average flow increase due to the leak
- Night flow is done by just checking differenece in previous aveerage nighttime flow measurements, has same treshold problem
- Window size can be adjusted to tweak the FPR and TPR
- uses interval of 5 min
- Data sanitazation is done through tresholding
- Increasing the number of fourier terms increases learning performance up to a certain point
- Results:
    - Approximation method uses parameters for quick learning, not optimal accuricy 
    - Threshold set to not give any FP
    - Has slow detection, but high accuricy. Detection speed can be increased by changing learning rate and detection treshold
    - Shows graph showing problem with night-flow analysis
    - Has worse accuricy, but quickers detection
    - Adaptive learning isn't affected by leakages that isn't detected within the time window(but can still only detect new leakages) 

## Summary:
- Current solution; Measure at DMA inlets and outlets and detection using treshold or manual operator observation
- Mentions the problems of summer and winter time
- AI application in leak fault detection:
    - Fuzzy min-max NN for pattern recognition
    - ANN for leak location and magnitude detection using pressure and flow
    - Fuzzy interface for confidence intervals(in leak detection)
    - GA for calibrating models
    - Kalman filters


#### WHO, gives examples of damage from leakage
url: https://apps.who.int/iris/bitstream/handle/10665/66893/WHO_SDE_WSH_01.1_eng.pdf?sequence=1
