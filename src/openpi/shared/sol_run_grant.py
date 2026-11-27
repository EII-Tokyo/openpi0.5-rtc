"""Short-lived paid-inference authorization shared by web, robot and policy."""
KEY = "aloha:rinse:sol_run_grant"
TTL_SECONDS = 12
CALIBRATION_KEY = "aloha:rinse:fixed_nozzle_tips"


def revoke(client, grant=None):
    if grant is None:
        client.delete(KEY)
    else:
        client.eval("if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) end",
                    1, KEY, grant)


def renew(client, grant):
    return bool(client.eval(
        "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('EXPIRE', KEYS[1], ARGV[2]) end",
        1, KEY, grant, TTL_SECONDS))


def active(client, grant):
    return isinstance(grant, str) and bool(grant) and client.get(KEY) in (grant, grant.encode())
