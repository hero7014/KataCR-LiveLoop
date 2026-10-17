from scripts.adb_probe import is_missing_device_error


def test_missing_device_error_is_retryable():
    assert is_missing_device_error("error: device '127.0.0.1:16384' not found") is True
    assert is_missing_device_error("error: device offline") is True


def test_unrelated_adb_error_is_not_retryable():
    assert is_missing_device_error("error: closed") is False


if __name__ == "__main__":
    test_missing_device_error_is_retryable()
    test_unrelated_adb_error_is_not_retryable()
    print("adb_probe_check passed")
