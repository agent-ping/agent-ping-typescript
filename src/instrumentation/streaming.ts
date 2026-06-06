export interface StreamObserver<TChunk> {
  onChunk(chunk: TChunk): void;
  onDone(): void;
  onError(err: unknown): void;
}

export function wrapAsyncIterable<TChunk>(
  source: AsyncIterable<TChunk>,
  observer: StreamObserver<TChunk>,
): AsyncIterable<TChunk> {
  return {
    [Symbol.asyncIterator](): AsyncIterator<TChunk> {
      const inner = source[Symbol.asyncIterator]();
      let done = false;
      const finalize = (): void => {
        if (done) return;
        done = true;
        try {
          observer.onDone();
        } catch {
          // swallow
        }
      };
      return {
        async next(): Promise<IteratorResult<TChunk>> {
          try {
            const result = await inner.next();
            if (result.done) {
              finalize();
              return result;
            }
            try {
              observer.onChunk(result.value);
            } catch {
              // swallow
            }
            return result;
          } catch (err) {
            try {
              observer.onError(err);
            } catch {
              // swallow
            }
            finalize();
            throw err;
          }
        },
        async return(value): Promise<IteratorResult<TChunk>> {
          finalize();
          if (inner.return) return inner.return(value);
          return { done: true, value: value as TChunk };
        },
        async throw(err): Promise<IteratorResult<TChunk>> {
          try {
            observer.onError(err);
          } catch {
            // swallow
          }
          finalize();
          if (inner.throw) return inner.throw(err);
          throw err;
        },
      };
    },
  };
}
