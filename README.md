# Curious AFL

This is the code base for the thesis "Curiosity Guided Fuzz Testing".
  
CuriousAFL extends [AFL](https://github.com/mirrorer/afl) with 3 different modes: MUTATION, RANDOM and CASE (undocumented in thesis). MUTATION and RANDOM extend afl-fuzz.c in function "common_fuzz_stuff" and CASE in function "calculate_score".

We extended AFL to use 3 additional flags:

| **AFL-Flag** | **Mode**   |
| --- | --- |
| -R MODE | Run CuriousAFL in either MUTATION (default), RANDOM, CASE   |
| -P **Port**| Connect to Python Curiosity RPC server on port **Port**. Only needed in MUTATION and CASE. |
| -r Percentile | If CuriousAFL runs in RANDOM mode, provide a percentile value to cancel out of seeds (e.g. 10, default: 0) |

## Seeds  
All seeds used in the thesis can be found in the [folder](https://github.com/derdav3/CuriousAFL/tree/master/seeds_programs) "seeds_programs".

## Examples
It's helpful to understand how to run "vanilla" AFL, before trying to run CuriousAFL. Please refer to the official [source](http://lcamtuf.coredump.cx/afl/README.txt) for a guide.

After following our [Installation guide](https://github.com/derdav3/CuriousAFL/wiki/Installation), the following are possible usecases (assuming you have CuriousAFL in /home/CuriousAFL and there is a seed folder `afl_in`):

### MUTATION (objdump)
`cd` into the testcase folder and start 2 terminals.  

Launch the python RND script:  
`python3 /home/CuriousAFL/CuriousAFL/rnd_server.py --projectbase=./ --port 44444`

Launch CuriousAFL:  
`/home/CuriousAFL/CuriousAFL/afl-fuzz -i afl_in/ -o afl_out/ -R MUTATION -P 44444 ./objdump -D @@`

### RANDOM (objdump)
`cd` into the testcase folder.  
(python script is not needed)

Launch CuriousAFL:  
`/home/CuriousAFL/CuriousAFL/afl-fuzz -i afl_in/ -o afl_out/ -R RANDOM -r 10 ./objdump -D @@`

