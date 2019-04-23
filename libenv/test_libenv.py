import os
import platform
import shlex
import subprocess
import contextlib
import shutil

import numpy as np
import pytest

from . import CVecEnv, scalar_adapter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_ENVS = ["ctestenv", "guess-number", "gotestenv"]


def setup_module(module):
    for env_name in TEST_ENVS:
        build(env_name)


@contextlib.contextmanager
def chdir(newdir):
    curdir = os.getcwd()
    try:
        os.chdir(newdir)
        yield
    finally:
        os.chdir(curdir)


def shell(cmd):
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)
    subprocess.check_call(cmd)


def build(env_name):
    """
    Build each test environment
    """
    env_dir = f"envs/{env_name}"
    build_dir = os.path.join(SCRIPT_DIR, "..", env_dir, "build")
    for build_type in ["Debug", "RelWithDebInfo"]:
        output_dir = os.path.join(build_dir, build_type)
        os.makedirs(output_dir, exist_ok=True)
        with chdir(output_dir):
            if platform.system() == "Linux":
                library_name = "libenv.so"
            elif platform.system() == "Darwin":
                library_name = "libenv.dylib"
            elif platform.system() == "Windows":
                library_name = "env.dll"
            else:
                raise Exception("unrecognized platform")

            if env_name in ["ctestenv", "guess-number"]:
                if not os.path.exists("CMakeCache.txt"):
                    generator = "Unix Makefiles"
                    if platform.system() == "Windows":
                        generator = "Visual Studio 15 2017 Win64"
                    shell(
                        f'cmake ../.. -G "{generator}" -DCMAKE_BUILD_TYPE={build_type}'
                    )
                # visual studio projects have to have the debug/release thing specified at build time rather than configure time
                # specify it both places just in case
                shell(f"cmake --build . --config {build_type}")
                if platform.system() == "Windows":
                    # on windows only, cmake seems to place the output files in a subfolder
                    shutil.copyfile(
                        os.path.join(build_type, library_name), library_name
                    )
            elif env_name in ["gotestenv"]:
                os.environ["GOPATH"] = os.path.dirname(build_dir)
                subprocess.check_call(
                    ["go", "build", "-o", library_name, "-buildmode=c-shared", env_name]
                )
            else:
                raise Exception("unrecognized environment name")


def get_lib_dir(env_name, debug):
    return os.path.join(
        SCRIPT_DIR,
        "..",
        "envs",
        env_name,
        "build",
        "Debug" if debug else "RelWithDebInfo",
    )


class CTestVecEnv(CVecEnv):
    def __init__(self, num_envs=1, debug=False, **kwargs):
        super().__init__(
            num_envs=num_envs,
            lib_dir=get_lib_dir("ctestenv", debug=debug),
            c_func_defs=["int special_function(int);"],
            debug=debug,
            **kwargs,
        )

    def special_function(self, x):
        return self.call_func("special_function", x)


CTestEnv = scalar_adapter(CTestVecEnv)


class GuessNumberVecEnv(CVecEnv):
    def __init__(self, num_envs=1, debug=False, **kwargs):
        super().__init__(
            lib_dir=get_lib_dir("guess-number", debug=debug),
            num_envs=num_envs,
            debug=debug,
            **kwargs,
        )


GuessNumberEnv = scalar_adapter(GuessNumberVecEnv)


class GoTestVecEnv(CVecEnv):
    def __init__(self, num_envs=1, debug=False, **kwargs):
        super().__init__(
            lib_dir=get_lib_dir("gotestenv", debug=debug),
            num_envs=num_envs,
            debug=debug,
            **kwargs,
        )


GoTestEnv = scalar_adapter(GoTestVecEnv)


def make_env(name, **kwargs):
    cls = {
        "ctestenv": CTestEnv,
        "guess-number": GuessNumberEnv,
        "gotestenv": GoTestEnv,
    }[name]
    return cls(**kwargs)


def make_venv(name, **kwargs):
    cls = {
        "ctestenv": CTestVecEnv,
        "guess-number": GuessNumberVecEnv,
        "gotestenv": GoTestVecEnv,
    }[name]
    return cls(**kwargs)


@pytest.mark.parametrize("num_envs", [1, 16])
@pytest.mark.parametrize("name", TEST_ENVS)
def test_envs_work(num_envs, name):
    """
    Make sure each environment works in debug mode
    """
    options = {}
    if name == "guess-number":
        options = {"num_bits": 32}
    venv = make_venv(name, debug=True, num_envs=num_envs, options=options)
    venv.reset()
    for _ in range(100):
        actions = [venv.action_space.sample() for _ in range(venv.num_envs)]
        actions = np.array(actions, dtype=venv.action_space.dtype)
        venv.step(actions)


@pytest.mark.parametrize("num_envs", [1, 16])
def test_ctestenv(num_envs):
    """
    Make sure ctestenv produces the correct output
    """
    options = None
    venv = make_venv("ctestenv", debug=True, num_envs=num_envs, options=options)
    env = make_env("ctestenv", debug=True, options=options)
    obs = venv.reset()
    scalar_obs = env.reset()
    steps = 0
    for _ in range(100):
        for i in range(num_envs):
            assert (obs["uint8_obs"][i].reshape(-1) == np.arange(1 * 2 * 3)).all()
            assert (obs["int32_obs"][i].reshape(-1) == np.arange(4 * 5 * 6)).all()
            assert (obs["float32_obs"][i].reshape(-1) == np.arange(7 * 8 * 9)).all()
            for key in obs.keys():
                assert obs[key].dtype is venv.observation_space.spaces[key].dtype
        for key in obs.keys():
            assert (obs[key][0] == scalar_obs[key]).all()
        actions = [
            np.ones_like(venv.action_space.sample()) * env_idx
            for env_idx in range(venv.num_envs)
        ]
        actions = np.array(actions, dtype=venv.action_space.dtype)

        venv.step_async(actions)
        obs, rews, dones, infos = venv.step_wait()
        scalar_obs, scalar_rew, scalar_done, scalar_info = env.step(actions[0])
        steps += 1

        assert (rews == np.arange(num_envs) * steps).all()
        assert scalar_rew == rews[0]
        frames = venv.get_images()
        scalar_frame = env.render(mode="rgb_array")
        for env_idx in range(num_envs):
            assert (frames[env_idx].reshape(-1) == steps * env_idx).all()
            assert infos[env_idx]["info"] == steps * env_idx
            if env_idx == 0:
                assert (frames[env_idx] == scalar_frame).all()
                assert (infos[env_idx]["info"] == scalar_info["info"]).all()

        if dones.any():
            break

    assert dones.all() and scalar_done and steps == 100
    venv.close()


def test_debug_state():
    """
    Make sure that calling methods out of order results in an assertion in debug mode
    """
    venv = make_venv("ctestenv", debug=True, num_envs=1, options=None)
    with pytest.raises(AssertionError):
        venv.step_wait()


def test_guess_number():
    """
    Make sure the guess-number environment mostly seems to work
    """
    num_envs = 3
    n = np.array([9999, 10000, 10001], dtype=np.int32)
    venv = GuessNumberVecEnv(debug=True, num_envs=num_envs, options={"n": n})

    def get_steps(nums):
        _obs = venv.reset()
        dones = np.array([False])
        step_count = 0
        while not dones.all():
            actions = np.array(
                [venv.action_space.sample() for _ in range(num_envs)],
                dtype=venv.action_space.dtype,
            )
            for i in range(num_envs):
                actions[i][0] = nums[i] & 0x1
                nums[i] >>= 1
            venv.step_async(actions)
            _obs, _rews, dones, _infos = venv.step_wait()
            assert dones.all() == dones.any(), "one environment finished before others"
            step_count += 1
        return step_count

    assert get_steps([i for i in n]) == 64
    assert get_steps([i - 1 for i in n]) == 1


@pytest.mark.parametrize("name", TEST_ENVS)
def test_env_speed(name, benchmark):
    if name == "guess-number":
        options = {"num_bits": 1024}
    else:
        options = {}
    venv = make_venv(name, options=options, reuse_arrays=True)
    actions = np.array([venv.action_space.sample()])

    def rollout(max_steps):
        venv.reset()
        step_count = 0
        while step_count < max_steps:
            venv.step_async(actions)
            venv.step_wait()
            step_count += 1

    benchmark(lambda: rollout(1000))


def test_define_func():
    env = CTestEnv()
    assert env.special_function(1337) == 1337
