# Disable ROS pytest plugins that fail on import in this Python 3.11 environment
# because they are built for Python 3.12 and require the 'lark' package.
def pytest_configure(config):
    for plugin in ("launch_pytest", "launch_testing"):
        try:
            config.pluginmanager.set_blocked(plugin)
        except Exception:
            pass
