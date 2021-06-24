![スクリーンショット 2021-05-12 10 47 07](https://user-images.githubusercontent.com/56621409/117906193-6a0c3b80-b30f-11eb-897a-55d6f74bb145.png)
# kaggle2021-CommonLit-Readability-Prize
https://www.kaggle.com/c/commonlitreadabilityprize コンペのリポジトリ
## Task
機械学習は、文章の適切な読解レベルを特定し、学習意欲を高めることができるのか？読書は、学業の成功に不可欠なスキルです。適切なレベルの課題を提供する魅力的な文章に触れることができれば、生徒は自然にリーディングスキルを身につけることができます。

現在、ほとんどの教育用テキストは、伝統的な読みやすさの方法や市販の計算式を使って読者に合わせています。しかし、それぞれに問題があります。Flesch-Kincaid Grade Levelのようなツールは、テキストのデコーディング（単語あたりの文字数や音節数など）や構文の複雑さ（文章あたりの単語数など）の弱い指標に基づいている。そのため、構成要素や理論的妥当性に欠けています。また、Lexileのように市販されている計算式は、コストが高く、適切な検証研究が行われておらず、計算式の特徴が公開されていないため、透明性の問題があります。

CommonLit, Inc.は、非営利の教育技術団体で、2,000万人以上の教師と生徒に、3年生から12年生までの無料のデジタルリーディングとライティングのレッスンを提供しています。アトランタにあるR1の公立研究大学であるジョージア州立大学と共同で、読みやすさの評価方法を改善するためにカグラーに挑戦しています。

このコンペティションでは、3年生から12年生のクラスで使用するために、読み物の複雑さを評価するアルゴリズムを構築します。そのためには、様々な年齢層の読者と、様々な分野から集められた大量のテキストを含むデータセットに、自分の機械学習スキルを組み合わせます。受賞モデルには、テキストのまとまりとセマンティクスが必ず組み込まれます。

成功すれば、管理者、教師、生徒の助けになるでしょう。パッセージを選ぶリテラシーカリキュラムの開発者や教師は、教室で使う作品を迅速かつ正確に評価できるようになります。さらに、これらの計算式は誰もが利用しやすくなります。おそらく最も重要なことは、生徒が自分の作品の複雑さや読みやすさについてのフィードバックを受けることができ、本質的なリーディングスキルをはるかに容易に向上させることができるということです。

## evaluation
投稿作品は、平均平方根誤差（RMSE）で採点されます。RMSEは次のように定義されます。
![スクリーンショット 2021-05-12 10 49 54](https://user-images.githubusercontent.com/56621409/117906406-ccfdd280-b30f-11eb-8903-0bb7a26c5020.png)

### 提出ファイル
```
id,target
eaf8e7355,0.0
60ecc9777,0.5
c0f722661,-2.0
etc.
```

## Result
  - public: x.xxx
  - private: x.xxx
  - rank: xx/xx

 <img src='' width='500'>

  
## Setting
* directory tree
```
kaggle2021-Molecular-Translation
├── README.md
├── data         <---- gitで管理するデータ
├── data_ignore  <---- .gitignoreに記述されているディレクトリ(モデルとか、特徴量とか、データセットとか)
├── nb           <---- jupyter lab で作業したノートブック
├── nb_download  <---- ダウンロードした公開されているkagglenb
└── src          <---- .ipynb 以外のコード
```
## Info
- [issue board](https://github.com/Hiroki29/kaggle2021-Molecular-Translation/projects/1)  

## train.csv colomn infomaiton
2834件
|name|Explanation|
|----|----|
|id|一意のID|
|url_legal|URLのソース|
|license|素材のライセンス|
|excerpt|読みやすさを予測するテキスト|
|target|読みやすさ|
|standard_error|各抜粋の複数の評価者間のスコアの広がりの尺度|

## test.csv colomn infomaiton
|name|Explanation|
|----|----|
|id|一意のID|
|url_legal|URLのソース|
|license|素材のライセンス|
|excerpt|読みやすさを予測するテキスト|


## Log
### 20210512
Kaggle日記始動

### 20210604
本格的に始める
#### 参考になりそうなEDA
* [CommonLit Readability Prize: EDA + Baseline](https://www.kaggle.com/ruchi798/commonlit-readability-prize-eda-baseline)
* [I.CommonLit: Explore + XGBRF&RepeatedFold Model](https://www.kaggle.com/andradaolteanu/i-commonlit-explore-xgbrf-repeatedfold-model)
![スクリーンショット 2021-06-04 10 43 55](https://user-images.githubusercontent.com/56621409/120733255-c551d800-c521-11eb-9cb1-21fd40a22570.png)
![スクリーンショット 2021-06-04 10 44 32](https://user-images.githubusercontent.com/56621409/120733301-dbf82f00-c521-11eb-8c2a-f720993c2c12.png)
![スクリーンショット 2021-06-04 10 45 16](https://user-images.githubusercontent.com/56621409/120733355-f5997680-c521-11eb-80bf-e9790eabd752.png)

#### 所感
* 非常にシンプルなタスクなため差が付きにくそう
* まずは一つのモデルをしっかりとさせることに注力する！！！
* アンサンブルは二の次である
* 様々なBertのアンサンブルが良さそう

### 20210611 
* まずはアンサンブルより単一なモデルの精度向上を目指す
* Robertaの仕組みがどうなっているのかを理解する
* Bertの新たな知見
	* MLM,NSPに加えて同じドメインでの事前学習行う![スクリーンショット 2021-06-11 5 42 24](https://user-images.githubusercontent.com/56621409/121594277-cfbb2700-ca77-11eb-88e8-11949ceed9ca.png)

* 新たな良さげnotebook
	* [Speeding up Transformer w/ Optimization Strategies](https://www.kaggle.com/rhtsingh/speeding-up-transformer-w-optimization-strategies)
	* [CommonLit Readability Prize - RoBERTa Torch|ITPT](https://www.kaggle.com/rhtsingh/commonlit-readability-prize-roberta-torch-itpt)
	* [BERT - In Depth Understanding](https://www.kaggle.com/mdfahimreshm/bert-in-depth-understanding)
![スクリーンショット 2021-06-11 6 06 48](https://user-images.githubusercontent.com/56621409/121597071-3726a600-ca7b-11eb-95f4-76c63e0ee147.png)
* なぜマルチヘッドアテンションを使うのか？
	* Self-Attentionでは, ある単語が単独で, 他の単語との注意スコアよりもはるかに多くの注意を引くことが観察される
	* これは, モデルが文脈を理解するのを妨げる可能性がある. そこで, self-Attentionのスコアを複数回測定することで, この問題を少しでも減らすことができる
* まずはしっかりスクラッチで自分のコードを書く,その際コピペはせずちゃんと書く
* 1実験1スクリプトで行う！参考（荒井さん）
	* https://speakerdeck.com/koukyo1994/konpezhong-falsekodo-dousiteru?slide=7
	![スクリーンショット 2021-06-11 11 26 57](https://user-images.githubusercontent.com/56621409/121621852-f1cc9d80-caa7-11eb-89ff-1415cc37f173.png)

### 20210612
実際にinferenceして提出してみた  
* roberta-base score : 0.517
* roberta-large score : 0.594
#### 気づき
* 作成したモデルはubuntuPCからkaggleに保存したほうが良い
* 提出するときはinternet接続できないからinputに入れておかないといけない
#### これからすること
* 学習率についての理解を深める
* epoch数はこのままでいいのか
* max_lengthについて
* アンサンブルの仕方（単に結果の平均を取るのか？）
* 最後に単なるlinearでよいのか
* 過去に優勝したNLPコンペをまとめる必要ありそう
* ![スクリーンショット 2021-06-12 10 49 43](https://user-images.githubusercontent.com/56621409/121761627-e93d9b00-cb6b-11eb-91e6-928a483601d0.png)
	* I calculated the validation score for every 40 steps (batch size: 8) and saved the model with the best validation score.(40ステップ（バッチサイズ：8）ごとに検証スコアを計算し、最高の検証スコアでモデルを保存しました。)
	* per 20 steps. 20ステップなどで評価するようにする![スクリーンショット 2021-06-12 10 51 57](https://user-images.githubusercontent.com/56621409/121761689-3883cb80-cb6c-11eb-8465-23f4b30d523f.png)
* I'm using a batch size 8![スクリーンショット 2021-06-12 10 53 16](https://user-images.githubusercontent.com/56621409/121761726-68cb6a00-cb6c-11eb-8cec-8e3f7b24e348.png)

#### 学習率について
* 現状 
```
AdamW (
Parameter Group 0
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 1e-05
    weight_decay: 0.003

Parameter Group 1
    betas: (0.9, 0.999)
    correct_bias: True
    eps: 1e-06
    lr: 1e-05
    weight_decay: 0.0
)
```

### 20210612
#### 今日やること
[My Experience So Far and Further Improvements](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/241029)
このノートブックが非常に有益である
![スクリーンショット 2021-06-13 6 27 21](https://user-images.githubusercontent.com/56621409/121789383-6c183180-cc10-11eb-8596-5dcb59eabe1f.png)
* Further Improvements - There are many ways to improve the score further and I will be implementing some and sharing them very soon.

		1. Dynamic Padding / Sequence Bucketing
		2. Stochastic Weight Averaging(確率的重み平均)
		3. Utilizing Different Layers - There are many ways to do this concat, mean pooling, max pooling, attention, lstm, hierarchical or parallel aggregation.
		4. Unsupervised Data Augmentation
		5. MIXOUT - At each training iteration, each model parameter is replaced with its pre-trained value with probability p.  
		https://arxiv.org/abs/1909.11299  論文
		6. Pretraining + Multi-Task Training + Fine-tuning(複数のタスクを同時に微調整) - Multi-Task Learning : Finetuning on multiple tasks simultaneously. Using OneStop English and Weebit Corpus.
		7. Pretraining + STILTs Training(複数のタスクを順番に微調整) - STILTs Training: Finetuning on multiple tasks sequentially. Using OneStop English and Weebit Corpus.
		8. LLRD(最上層には高い学習率を適用し、最下層には低い学習率を適用) - Layer Wise Learning Rate Decay - Introduced in (Howard & Ruder, 2018). Applies higher learning rates for top layers and lower learning rates for bottom layers. The 4th notebook which does something similar where instead grouping of layers is done.
		9. Training Encoder and Head Separately(エンコーダーとヘッドを別々にトレーニングする?)
		10. Backtranslation Augmentation(逆翻訳を用いたデータの拡張)
		11. Larger and More Models(他の大きなモデル) - RoBERTa Large, XLNet, T5, Electra, Albert these models are comparatively trained on larger datasets than other models and thus might perform better.
		12. Ensembling Techniques(アンサンブル) - Like every other Kaggle competition this will be the key.

* 前処理について  
You can see example code here: https://github.com/moizsaifee/kaggle-jigsaw-multilingual-toxic-comment-classification-3rd-place-solution

### 20210613
#### 行ったこと
* discussionを眺めた
	* 前処理とfineturningの2段処理をするべき
	* 単一モデルについて　20step 
* 実際に提出した
	*https://www.kaggle.com/rhtsingh/commonlit-readability-prize-roberta-torch-infer-3 これをもとにした	 
### 20210614
#### やることリスト
* SVMの勉強
* trainでinferenceをできるようにする infer/など？
* アンサンブルの仕方検証
#### 発見
* 新しいcode
	* [On Stability of Few-Sample Transformer Fine-Tuning](https://www.kaggle.com/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning)
	* [CommonLit Readability2 - 日本語](https://www.kaggle.com/tensorchoko/commonlit-readability2) roberta+svm で PB0.465
* traindataに対してのスコアを出せるようにする infer ディレクトリの作成
* bert出力後にattentionheadをつけてみるとどうなるのか
* attention head の埋め込み後にsvmを使用してPB0.474
	* https://www.kaggle.com/maunish/clrp-roberta-svm

### 20210615
#### やることリスト
* 結果のまとめの作成
* SVMを用いることでの性能向上？
* 以下を試したい
* https://www.kaggle.com/rhtsingh/on-stability-of-few-sample-transformer-fine-tuning
	* Debiasing Omission In BertADAM
	* Re-Initializing Transformer Layers
	* Utilizing Intermediate Layers
	* Layer-wise Learning Rate Decay (LLRD)
	* Mixout Regularization
	* Pre-trained Weight Decay
	* Stochastic Weight Averaging

### 20210624
#### やることリスト
* 他のデータの切り方で試す
	* https://www.kaggle.com/shahules/guide-pytorch-data-samplers-sequence-bucketing 	
* [Utilizing Transformer Representations Efficiently](https://www.kaggle.com/c/commonlitreadabilityprize/discussion/247615)
	* pooler embedingを使用するだけでは少ない、様々な層の中間表現を利用
		* Pooler Output
		* Last Hidden State Output
		* Hidden Layers Output
* モデルだけでなく原点に立ち返ってデータから分かることも有るはず
* 他の自然言語処理のタスクからも応用できることを探すべき
* 前処理、後処理について考える必要あり
	* 前処理:	 
	* 後処理:
	  	
### 20210625
#### やることリスト
* ゼロ作で最適化、BNについて復習　コード動かす
* schedulerについて勉強
* layer毎にschedulerどのように変えているのかを調べる
