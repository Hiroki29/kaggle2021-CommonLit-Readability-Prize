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
notebook: nb001
example: https://www.xeno-canto.org/134874

|name|Explanation|
|----|----|
|image_id|画像のID|
|InChI|国際化学物質識別番号()|

![スクリーンショット 2021-05-12 8 49 27](https://user-images.githubusercontent.com/56621409/117898028-f6fac900-b2fe-11eb-8141-42a68c20709e.png)


## Log
### 20210512
Kaggle日記始動

### 202105xx
Dockerで環境構築
