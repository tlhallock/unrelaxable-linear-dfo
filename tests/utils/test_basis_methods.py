from ludfo.utils.basis import get_quadratic_basis_dim


def test_basis_dim_2():
    assert get_quadratic_basis_dim(2) == 6




# if __name__ == '__main__':
#     print(vandermonde(np.random.random((3, 2))))
#     # for b in quadratic_basis(3):
#         # print(b)