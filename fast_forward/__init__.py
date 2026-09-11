# Copyright 2020 University of Groningen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from importlib.metadata import PackageNotFoundError, version

# try:
#     __version__ = version('fast_forward')
# except PackageNotFoundError:
#     __version__ = '0+unknown'

# Find the data directory once.
try:
    from importlib.resources import files, as_file
    import atexit
    from contextlib import ExitStack
except ImportError:
    from pathlib import Path
    DATA_PATH = Path(__file__).parent / 'data'
    del Path
else:
    ref = files('fast_forward') / 'data'
    file_manager = ExitStack()
    atexit.register(file_manager.close)
    DATA_PATH = file_manager.enter_context(as_file(ref))
    del files, as_file, atexit, ExitStack


# import numba if available
import functools
try:
    from numba import jit
except:
    jit = lambda x: x
    print("INFO - Couldn't import numba. Install it for a speed boost.")
else:
    jit = functools.partial(jit,  nopython=True, cache=True, fastmath=True)
