This repository contains the code used in:

Hitczenko, K., Mazuka, R., Elsner, M., & Feldman, N. H. (2020). "When context is and isn't helpful: A corpus study of naturalistic speech." Psychonomic Bulletin and Review, 27(4), 640-676.

The contents consist of:
- normtopdown.R: implements the logistic regression normalization model as well as logistic regression top-down information model
- normalize_nn.py: implements the neural network normalization model.

For both scripts, the input data is structured such that each line contains information about one vowel in the corpus:
length, duration, f1, f2, f3, contextual_factor_1, contextual_factor_2,...,contextual_factor_n

We ran our analyses on the R-JMICC and Werker data sets. See the following papers for more information about these corpora. Their distribution is controlled by the authors/creators.

Mazuka, Reiko, Yosuke Igarashi, and Ken'ya Nishikawa. "Input for learning Japanese: RIKEN Japanese mother-infant conversation corpus." 電子情報通信学会技術研究報告. TL, 思考と言語 106.165 (2006): 11-15.

Werker, J. F., Pons, F., Dietrich, C., Kajikawa, S., Fais, L., & Amano, S. (2007). Infant-directed speech supports phonetic category learning in English and Japanese. Cognition, 103(1), 147-162.

Please do not hesitate to reach out to Kasia Hitczenko with any questions!
