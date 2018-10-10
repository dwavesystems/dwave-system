"""
    This test is written for command line handling.

    For a starting place just tries to check if all six of the major entry
    points get reached in very simple situations.
"""

input_data = """
0 4
0 5
0 6
0 7
1 4
1 5
1 6
1 7
2 4
2 5
2 6
2 7
3 4
3 5
3 6
3 7
"""

if __name__ == '__main__':
    import runpy
    import mock
    import sys
    import os
    sys.path.append(os.path.abspath('bin'))
    sys.path.append(os.path.abspath('src'))
    from chimera_embedding import processor

    # Largest native clique with no other options given
    proc = mock.MagicMock(processor)
    m = mock.mock_open(read_data=input_data)
    try:
        with mock.patch('chimera_embedding.processor', proc):
            with mock.patch('__main__.open', m) as mock_open:
                with mock.patch('argparse.open', m) as mock_open:
                    sys.argv = [sys.argv[0]] + '-i test_file.dat -o not_the_same_name.dat'.split(' ')
                    runpy.run_module('nativeclique', run_name='__main__')
    except Exception as error:
        # print error
        pass

    m.assert_any_call('test_file.dat', 'r', -1)
    m.assert_any_call('not_the_same_name.dat', 'w', -1)
    assert(mock.call().largestNativeClique() in proc.mock_calls)

    # Call nativeCliqueEmbed with chain length argument - 1
    proc.reset_mock()
    m.reset_mock()
    try:
        with mock.patch('chimera_embedding.processor', proc):
            with mock.patch('__main__.open', m) as mock_open:
                with mock.patch('argparse.open', m) as mock_open:
                    sys.argv = [sys.argv[0]] + '-i test_file.dat -o not_the_same_name.dat --chainlength=10'.split(' ')
                    runpy.run_module('nativeclique', run_name='__main__')
    except Exception as error:
        # print error
        pass

    m.assert_any_call('test_file.dat', 'r', -1)
    m.assert_any_call('not_the_same_name.dat', 'w', -1)
    assert(mock.call().nativeCliqueEmbed(9) in proc.mock_calls)

    # Call tightestNativeClique with clique size provided
    proc.reset_mock()
    m = mock.mock_open(read_data=input_data)
    try:
        with mock.patch('chimera_embedding.processor', proc):
            with mock.patch('__main__.open', m) as mock_open:
                with mock.patch('argparse.open', m) as mock_open:
                    sys.argv = [sys.argv[0]] + '-i test_file.dat -o not_the_same_name.dat --cliquesize 4'.split(' ')
                    runpy.run_module('nativeclique', run_name='__main__')
    except Exception as error:
        # print error
        pass

    m.assert_any_call('test_file.dat', 'r', -1)
    m.assert_any_call('not_the_same_name.dat', 'w', -1)
    assert(mock.call().tightestNativeClique(4) in proc.mock_calls)

    # Call largestNativeBiClique
    proc.reset_mock()
    m = mock.mock_open(read_data=input_data)
    try:
        with mock.patch('chimera_embedding.processor', proc):
            with mock.patch('__main__.open', m) as mock_open:
                with mock.patch('argparse.open', m) as mock_open:
                    sys.argv = [sys.argv[0]] + '-i test_file.dat -o not_the_same_name.dat --bipartite'.split(' ')
                    runpy.run_module('nativeclique', run_name='__main__')
    except Exception as error:
        # print error
        pass

    m.assert_any_call('test_file.dat', 'r', -1)
    m.assert_any_call('not_the_same_name.dat', 'w', -1)
    assert(mock.call().largestNativeBiClique() in proc.mock_calls)

    # Call tightestNativeBiClique
    proc.reset_mock()
    m = mock.mock_open(read_data=input_data)
    try:
        with mock.patch('chimera_embedding.processor', proc):
            with mock.patch('__main__.open', m) as mock_open:
                with mock.patch('argparse.open', m) as mock_open:
                    sys.argv = [sys.argv[0]] + '-i test_file.dat -o not_the_same_name.dat --bipartite --cliquesize 4'.split(' ')
                    runpy.run_module('nativeclique', run_name='__main__')
    except Exception as error:
        # print error
        pass

    m.assert_any_call('test_file.dat', 'r', -1)
    m.assert_any_call('not_the_same_name.dat', 'w', -1)
    assert(mock.call().tightestNativeBiClique(4, m=4, chain_imbalance=None, max_chain_length=None) in proc.mock_calls)

    # Call tightestNativeBiClique again
    proc.reset_mock()
    m = mock.mock_open(read_data=input_data)
    try:
        with mock.patch('chimera_embedding.processor', proc):
            with mock.patch('__main__.open', m) as mock_open:
                with mock.patch('argparse.open', m) as mock_open:
                    sys.argv = [sys.argv[0]] + '-i test_file.dat -o not_the_same_name.dat --bipartite --cliquesize 4 7 --chainlength 80'.split(' ')
                    runpy.run_module('nativeclique', run_name='__main__')
    except Exception as error:
        # print error
        pass

    m.assert_any_call('test_file.dat', 'r', -1)
    m.assert_any_call('not_the_same_name.dat', 'w', -1)
    assert(mock.call().tightestNativeBiClique(4, m=7, chain_imbalance=None, max_chain_length=80) in proc.mock_calls)


    # Call largestNativeBiClique
    proc.reset_mock()
    m = mock.mock_open(read_data=input_data)
    try:
        with mock.patch('chimera_embedding.processor', proc):
            with mock.patch('__main__.open', m) as mock_open:
                with mock.patch('argparse.open', m) as mock_open:
                    sys.argv = [sys.argv[0]] + '-i test_file.dat -o not_the_same_name.dat --bipartite --chainlength 6'.split(' ')
                    runpy.run_module('nativeclique', run_name='__main__')
    except Exception as error:
        # print error
        pass

    m.assert_any_call('test_file.dat', 'r', -1)
    m.assert_any_call('not_the_same_name.dat', 'w', -1)
    assert(mock.call().largestNativeBiClique(max_chain_length=6, chain_imbalance=None) in proc.mock_calls)
