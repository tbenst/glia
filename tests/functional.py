from conftest import assert_within
import glia

def zip_dictionaries_test():
	a = {i:"a" for i in range(5)}
	b = {i:"b"*i for i in range(3)}
	c = {i:"c" for i in range(4)}but

	x = glia.zip_dictionaries(a,b,c)

	assert list(x) == [(0, ('a', '', 'c')), (1, ('a', 'b', 'c')), (2, ('a', 'bb', 'c'))]
