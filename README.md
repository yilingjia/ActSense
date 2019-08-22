# ActSense

This is an implementation for the paper titled "Active Collaborative Sensing for Energy Breakdown" which is published at CIKM 2019. We public our source code in this repository.

### Algorithm
ActSense model amis at minimizing the deployment cost by selectively deploying sensing hardware to a subset of homes and appliances while maximizing the reconstruction accuracy of sub-metered readings in non-instrumented homes.
We perform this active sensor deployment via active tensor completion with streaming data. Specifically, at the end of each month, we query the home and appliance pairs that have the highest uncertainty in the current tensor reconstruction, which we prove to reduce reconstruction uncertainty mostly rapidly. And to project a model's prediction uncertainty of future readings in a longer term, we incorporate external seasonal information into model estimationm, which helps the model react to future season changes earlier.
The detailed algorithm can be found in the paper.

### Usage
To run the code to generate experimental results like those found in our papers, you will need to run a command in the following format, using Python 3:

#### For our proposed method ActSense
```
$ cd code
$ python active_sensing.py [--year] [--dataset] [--method] [--init] 
                           [--uncertainty] [--alpha1] [--alpha2] [--alpha3]
                           [--k] [--latent_dimension] [--season_type] 
                           [--regularization] [--gamma1] [--gamma2]
                           [--lambda1] [--lambda2] [--lambda3]
                           [--kernel] [--sigma]
```
#### For baseline: random selection
```
$ cd code
$ python random_selection.py [--year] [--dataset] [--init] 
                             [--k] [--latent_dimension] [--season_type] 
                             [--regularization]
                             [--lambda1] [--lambda2] [--lambda3]
```
#### For baseline: query by committee
```
$ cd code
$ python query_by_committee.py [--year] [--dataset] [--init] 
                               [--k] [--latent_dimension] [--season_type] 
                               [--regularization]
                               [--lambda1] [--lambda2] [--lambda3]
```
The results will be stored in ../data/result/

We use the [Dataport](https://www.pecanstreet.org/dataport/) dataset for evaluation purpose. It is thelargest public residential home energy dataset, which containsthe appliance-level and household aggregate energy consumptionsampled every minute from 2012 to 2018.

 We filter out the appliances with poor data quality (large proportion of missing values) to select a subset of them. We get 4 different datasets from year 2014 to 2017 containing 53, 93, 73, and 44 homes respectively and six appliances: air-conditioning (HVAC),fridge, washing machine, furnace, microwave and dishwasher. Onthis selected data set, we reconstruct the aggregate reading by thesum of the selected appliances 

### Citation
If you use this code to produce results for your scientific publication, please refer to our CIKM 2019 paper:
