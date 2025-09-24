
## DeCaL Reasoner Evaluation for Class Expression Learning

This experiment demonstrates how to use the **DeCaL** reasoner for class expression learning in Description Logics. The key idea is to show that **DeCaL** can work with **any concept learning algorithm**, and can adapt to different configurations controlled by parameters `p`, `q`, and `r`.



## Installation

```shell
# To create a virtual python env with conda 
conda create -n venv python=3.10.14 --no-default-packages && conda activate venv && pip install -e . && cd Ontolearn
# To unzip the benchmark datasets knowledge graphs
unzip KGs.zip
# To unzip the learning problems
unzip LPs.zip
```
Other datasets and learning problems can be manually downloaded from [here](https://drive.google.com/file/d/1LWmrtVQFh2_9eWOUsGZTGVeTkxi3n5pk/view?usp=sharing) 



## Goal 

To evaluate the robustness and flexibility of DeCaL as a backend reasoner when plugged into any concept learning system, here we use a learning pipeline applied to the Family dataset.


| Argument                | Description                                                     |
| ----------------------- | --------------------------------------------------------------- |
| `--max_runtime`         | Max runtime per learning problem (default: `10` seconds)        |
| `--lps`                 | Path to normal learning problems (`lps.json`)                   |
| `--lps_difficult`       | Path to more challenging problems                               |
| `--kb`                  | Path to the OWL knowledge base (e.g., the Family ontology)      |
| `--path_pretrained_kge` | Path to pretrained KGE embeddings (optional)                    |
| `--data_name`           | Name of the dataset (`family` by default)                       |
| `--reasoner`            | Reasoner used (`EBR`)                            |
| `--operation`           | Type of operation: `normal`, `incomplete`, or `inconsistent`    |
| `--use_cache`           | Whether to use a semantic cache (`True`/`False`)                |
| `--gamma`               | Threshold parameter for the EBR reasoner (default: `0.5`)       |
| `--p`, `--q`, `--r`     | Parameters for DeCaL                                            |

Other parameters which are internal to DeCaL have been fixed to:

`--embedding_dim`: 32 (Embedding dimension to represent nodes and entities)

`--num_epochs`: 100 Number of Epochs to train DeCaL

`--learning_rate`: 0.1

`--batch_size`: 1024


## Example Usage

To run the experiment with DeCaL and a specific parameter combination:

```bash
python examples/concept_learning_evaluation_reasoners.py \
  --reasoner --p 1 --q 1 --r 1
  ```

We use a simple triple-loop to run the learning script with all combinations of the DeCaL parameters `p`, `q`, and `r` in `{0, 1}`:

```bash
for p in 0 1; do
  for q in 0 1; do
    for r in 0 1; do
      python examples/concept_learning_evaluation_reasoners.py --p $p --q $q --r $r
    done
  done
done
```


This script runs the learning system 8 times, each with a different configuration of the DeCaL parameters (where p, q, and r vary over 0 and 1). These parameters influence how DeCaL reasons over the knowledge base during concept learning.



## Concept learning with EBR

To get the results on concept learning on the error-free Family dataset, run

```shell
python examples/concept_learning_evaluation_reasoners.py --reasoner Pellet --operation normal --kb "KGs/Family/family-benchmark_rich_background.owl" --lps "LPs/Family/lps.json"
```

This will run the algorithm of the four concept learners CELOE, OCEL, CLIP, and Evolearner with Pellet as the reasoner on the family dataset.

After the `--reaoner` flag, we can choose different other reasoners: `["EBR", "Pellet", "HermiT", "JFact", "Openllet", "Structural"]`

To have the results on the inconsistent or incomplete put after the  ```--operation```  argument `inconsistent` or `incomplete`.

The results for other datasets can be obtained in a similar manner by changing the knowledge base argument `--kb` and the corresponding learning problems `--lps`.
The path to all knowledge bases can be found at `Ontolearn/KGs` and `Ontolearn/datasets` while the learning problems are in `Ontolearn/LPs`.
For instance the path to the Vicodi dataset is `Ontolearn/datasets/vicodi/kb` and the corresponding LPs can be found at `Ontolearn/datasets/vicodi/training_data/training_data_prep.json`

Therefore the result for the inconsistent Vicodi dataset with ratio 0.1 using the EBR reasoner can be obtained by running

```shell
python examples/concept_learning_evaluation_reasoners.py --reasoner EBR --operation inconsistent --ratio 0.1 --kb "datasets/vicodi/kb" --lps "datasets/vicodi/training_data/training_data_prep.json"
```

## Effect of the threshold

To see the effect of the threhold gamma, run the same codes by adding the argument `--gamma 0.9` which means we are setting a threshold of 0.9. The default threshold is set to 0.5


## Example of the concept learning results on the Family dataset

| LP  | F1-OCEL | RT-OCEL | F1-CELOE | RT-CELOE | F1-Evo | RT-Evo | F1-clip | RT-clip |
|-----|---------|---------|----------|----------|--------|--------|----------|----------|
| Grandson | 1.00000 | 0.15721 | 1.00000 | 0.00689 | 1.00000 | 0.03915 | 1.00000 | 0.00731 |
| PersonWithASibling | 1.00000 | 0.00344 | 1.00000 | 0.00228 | 1.00000 | 0.04062 | 1.00000 | 0.00256 |
| Uncle | 0.89412 | 62.44005 | 0.90476 | 11.06893 | 0.93827 | 0.08874 | 0.93827 | 60.14672 |
| Granddaughter | 1.00000 | 0.05480 | 1.00000 | 0.00551 | 1.00000 | 0.04122 | 1.00000 | 0.00782 |
| Brother ⊔ (∃ married.(Son ⊔ (∀ hasSibling.Parent))) | 0.95177 | 43.20808 | 0.94983 | 4.71822 | 0.90230 | 0.07865 | 0.94983 | 4.68857 |
| Brother ⊔ (∃ married.(Male ⊓ (∀ hasParent.(¬Person))))    |    0.87500    |   60.26158    |     0.90000     |   60.01146     |  0.86222     |  0.09792   |     0.90000   |    60.33988
|        Grandmother ⊔ (∀ hasSibling.Granddaughter)      |  0.96250   |    60.19428     |    0.96250    |    60.34919   |    0.90217   |    0.06330    |    0.96250   |    60.49495
|  Brother ⊔ (∃ married.(PersonWithASibling ⊓ (∀ hasSibling.Parent)))   |    0.67416    |   41.27145     |    0.67416     |   60.25687    |   0.67416    |   0.06932   |     0.67416    |   60.24825


