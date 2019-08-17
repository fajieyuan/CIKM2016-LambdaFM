Include Lambdafm (java) and Lambdafm (tensorflow)



LambdaFM has been used in many recommender systems, e.g., Tencent Inc: https://cloud.tencent.com/developer/article/1063837 
Note that LambdaFM (JAVA) code can be added in fBGD framework (https://github.com/fajieyuan/fBGD)  following the way of PRFM (Java file: RankFM_nextmusic). 

If you find the java code is very slow, you can refactor it in your own framework since the sampling in librec1.3 is very show.

@inproceedings{yuan2016lambdafm,
  title={Lambdafm: learning optimal ranking with factorization machines using lambda surrogates},
  author={Yuan, Fajie and Guo, Guibing and Jose, Joemon M and Chen, Long and Yu, Haitao and Zhang, Weinan},
  booktitle={Proceedings of the 25th ACM International on Conference on Information and Knowledge Management},
  pages={227--236},
  year={2016},
  organization={ACM}
}
