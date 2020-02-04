# A Coevolutionary Framework for Automatic Extraction of a Health Indicator from Vibrational Data using Sparse Autoencoders

### Installing the requirements:
  1. $ pip install setuptools --upgrade --ignore-installed
  2. $ pip install -r --upgrade requirements.txt

### Using the code:
  1. Open main.py
  2. Set the global random seeds (immediately after the import statements)
  3. Set the variables
  4. Run the code

### Outputs:
  1. Fitness values of networks at each generation are printed out
  2. Graphs are saved in "results/" directory
---
### An example:

#### Variables:

  * Numpy and Random library seeds = 0
  * nLayerSpecies = 4
  * popSizeBits = 3
  * netPopSize = 10
  * nGens = 3
  * iters = 100
  
#### Outputs:
Evaluating Initial Population...

Initial Population Fitness Values:

(11.373667953863205, 0.28914862871170044)  
(9.232717356954643, 0.1736980676651001)  
(9.606802505674105, 0.7317416667938232)  
(0.01, 100.0)  
(0.01, 100.0)  
(0.01, 100.0)  
(0.01, 100.0)  
(0.01, 100.0)  
(0.01, 100.0)  
(0.01, 100.0)

Generation:  1  
Elapsed Time:  0.8356954137484233  minutes

(11.50470342968584, 0.27457094192504883)  
(9.2115826027897, 0.173610121011734)  
(11.373667953863205, 0.28914862871170044)  
(9.141837914045395, 0.19460007548332214)  
(9.606802505674105, 0.7317416667938232)  
(7.349610760858365, 0.4088919281959534)  
(7.176305776705846, 0.9412201046943665)  
(0.01, 100.0)  
(0.01, 100.0)  
(0.01, 100.0)

Generation:  2  
Elapsed Time:  2.115152927239736  minutes

(12.479015596689639, 0.46278931498527526)  
(11.50470342968584, 0.27457094192504883)  
(9.2115826027897, 0.173610121011734)  
(11.373667953863205, 0.28914862871170044)  
(9.137610963212406, 0.19463995099067688)  
(11.033398411807648, 0.8416904211044312)  
(10.940405493481906, 0.4246766924858093)  
(9.606802505674105, 0.7317416667938232)  
(0.01, 100.0)  
(0.01, 100.0)  

Generation:  3  
Elapsed Time:  3.489908957481384  minutes

(12.479015596689639, 0.46278931498527526)  
(11.50470342968584, 0.27457094192504883)  
(10.532504738098536, 0.06718762964010239)  
(11.373667953863205, 0.28914862871170044)  
(9.663866341919446, 0.1964147984981537)  
(11.259540281372521, 0.389418363571167)  
(7.679312925831451, 0.3140382170677185)  
(8.14850446829315, 2.9297173023223877)  
(0.01, 100.0)  
(0.01, 100.0)  

Total run time is: 0.08188499616252051 hours
