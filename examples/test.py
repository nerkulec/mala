from tqdm.auto import trange

import signal

def sigusr1_handler(sig, frame):
  print('SIGUSR1 received, raising KeyboardInterrupt')
  raise KeyboardInterrupt

signal.signal(signal.SIGUSR1, sigusr1_handler)

def work(name='work'):
  print(name)
  w = 0
  for i in trange(10000000000):
    w += i*i
  return w


if __name__ == '__main__':
  try:
    work()
  except KeyboardInterrupt:
    print('Interrupted')
    work('Interrupted')
  else:
    print('Completed')

