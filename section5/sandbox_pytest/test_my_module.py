from my_module import square


def test_square_gives_correct_value(input_value):
    # when
    subject = square(input_value)

    # then
    assert subject == 16
