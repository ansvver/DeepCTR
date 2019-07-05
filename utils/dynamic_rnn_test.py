import  tensorflow as tf


import  numpy as np

def dynamic_rnn(rnn_type = 'lstm'):
    X = np.random.randn(3, 6, 4)

    X[2, 4:] = 0
    X_seqlengths = [6, 6, 4]

    rnn_hidden_size = 5

    if rnn_type == 'lstm':
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_hidden_size,
                                            state_is_tuple=True)
    else:
        cell = tf.nn.rnn_cell.GRUCell(num_units=rnn_hidden_size)

    outputs, last_stats = tf.nn.dynamic_rnn(cell = cell,
                                            dtype = tf.float64,
                                            sequence_length = X_seqlengths,
                                            inputs = X)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        output, stats = session.run([outputs, last_stats])
        print(np.shape(output))
        print(output)
        print('*****************************')
        print(np.shape(stats))
        print(stats)


if __name__ == '__main__':
    dynamic_rnn('lstm')