# Integrating Vectorized Lexical Constraints for Neural Machine Translation

## Prerequisites
This repository is based on [Fairseq](https://github.com/pytorch/fairseq). Please see [here](https://github.com/pytorch/fairseq) for the environment configuration.

## Usage
Suppose the local path to this repository is `CODE_DIR`.
### Step 1: Data Preparation
We firstly use [this script](https://github.com/ghchen18/cdalign/blob/main/scripts/extract_phrase.py) to extract the lexical constraints for the training, validation and test sets. After that, we prepend the constraints before the target sentences. The constraints are separated by a special token `<sep>`. We use `<pad>` as the placeholder for sentence pairs with less than three constraint pairs. Here is an example:
```
# Source corpus
您 对 此 有 何 看法 ?
截至 发@@ 稿 时 , 经过 各族 群众 和 部队 官兵 连续 20 多 个 小时 的 紧急 抢修 , 麻@@ 塔 公路 已 基本 恢复 通行 。
# Target corpus
<pad> <sep> <pad> <sep> <pad> <sep> <pad> <sep> <pad> <sep> <pad> <sep> what is you view on this matter ?
部队 官兵 <sep> troops <sep> 群众 <sep> mobilized <sep> 紧急 <sep> rush <sep> the ho@@ tian military sub - command mobilized more than 500 troops and civilians to rush repair the highway .
```
We then binarize the text corpora using the following command. Please refer to [here](https://github.com/pytorch/fairseq) for more details.
```bash
python $CODE_DIR/fairseq_cli/preprocess.py -s zh -t en \
    --joined-dictionary \
    --trainpref $trainpref \
    --validpref $validpref \
    --testpref $testpref \
    --destdir $data_bin \
    --workers 32
```
### Step 2: Train the Vanilla Transformer Model
We train the vanilla model using the following command.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python $CODE_DIR/fairseq_cli/train.py $data_bin \
    --target-key-sep $index_for_sep \
    --ls-segment-indices "0,1" --ls-segment-weights "1,1" \
    --fp16 --seed 32 --ddp-backend no_c10d \
    -s zh -t en \
    --lr-scheduler inverse_sqrt --lr 0.0007 \
    --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --max-update 50000 \
    --weight-decay 0.0 --clip-norm 0.0 --dropout 0.3 \
    --max-tokens 8192 --update-freq 1 \
    --arch transformer --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --save-dir $CKPTS \
    --tensorboard-logdir $LOGS \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --no-progress-bar --log-format simple --log-interval 10 \
    --no-epoch-checkpoints \
    --save-interval-updates 1000 --keep-interval-updates 5 \
    |& tee -a $LOGS/train.log
```
### Step 3: Integrate Vectorized Constraints
We then train the constraint-aware model based on the checkpoint of the vanilla model.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python $CODE_DIR/fairseq_cli/train.py $data_bin \
    --finetune-from-model $path_to_vanilla_ckpt \
    --target-kv-table --target-key-sep $index_for_sep \
    --ls-segment-indices "0,1" --ls-segment-weights "$beta,$alpha" \
    --lambda-rank-reg 0 \
    --kv-attention-dropout 0.1 --kv-projection-dropout 0.1 \
    --plug-in-type type2 --plug-in-forward bottom --plug-in-component encdec \
    --plug-in-project none --aggregator-v-project --plug-in-v-project --plug-in-k-project \
    --plug-in-mid-dim 512 \
    --lr-scheduler cosine --lr 1e-7 --max-lr 1e-4 \
    --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr-shrink 1 --lr-period-updates 6000 --max-update 10000 \
    --fp16 --seed 32 --ddp-backend no_c10d \
    -s zh -t en \
    --weight-decay 0.0 --clip-norm 0.0 --dropout 0.1 \
    --max-tokens 8192 --update-freq 1 \
    --arch transformer --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --save-dir $CKPTS \
    --tensorboard-logdir $LOGS \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --no-progress-bar --log-format simple --log-interval 10 \
    --no-epoch-checkpoints \
    --save-interval-updates 1000 --keep-interval-updates 1 \
    |& tee -a $LOGS/train.log
```
### Step 4: Generate with Constraints
We use the following command to translate with constraints.
```bash
CUDA_VISIBLE_DEVICES=0 python $CODE_DIR/generate.py $data_bin \
    --fp16 \
    -s zh -t en \
    --path $path_to_last_ckpt --gen-subset test \
    --beam 4 \
    --batch-size 128 \
    --target-key-sep $index_for_sep
```
## Citation
Please cite as:
```bibtex
@inproceedings{Wang:2022:VecConstNMT,
  title = {Integrating Vectorized Lexical Constraints for Neural Machine Translation},
  author = {Shuo Wang, Zhixing Tan, Yang Liu},
  booktitle = {Proceedings of ACL 2022},
  year = {2022},
```
## Contact
If you have questions, suggestions and bug reports, please email [wangshuo.thu@gmail.com](mailto:wangshuo.thu@gmail.com).