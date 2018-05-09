## basic_seq2seq_attn
the basic seq2seq with attention, it's a baseline of my experiment of neural text summarization. it was written based dataset CNN/Dailymail.

#where to find the dataset:
To obtain the CNN / Daily Mail dataset, follow the instructions [here](https://github.com/abisee/cnn-dailymail). Once finished, you should have [chunked](https://github.com/abisee/cnn-dailymail/issues/3) datafiles train_000.bin, ..., train_287.bin, val_000.bin, ..., val_013.bin, test_000.bin, ..., test_011.bin (each contains 1000 examples) and a vocabulary file vocab.


#train stage
run commmand:
python run_summarization.py --mode=train --data_path=the_data_path/chunked/train_* --vocab_path=the_data_path/vocab --log_root=the_log_path --exp_name=the_exp_path

#eval stage
run command:
python run_summarization.py --mode=eval --data_path=the_data_path/chunked/val_* --vocab_path=the_data_path/vocab --log_root=the_log_path --exp_name=the_exp_path

#decode stage
run command:
python run_summarization.py --mode=decode --data_path=the_data_path/chunked/test_* --vocab_path=the_data_path/vocab --log_root=the_log_path --exp_name=the_exp_path

if you want to get the final result and calculate the rouge, run this command:
python run_summarization.py --mode=decode --data_path=the_data_path/chunked/test_* --vocab_path=the_data_path/vocab --log_root=the_log_path --exp_name=the_exp_path --single_pass=True



