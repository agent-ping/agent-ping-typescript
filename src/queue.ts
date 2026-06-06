export interface QueueStats {
  size: number;
  dropped: number;
}

export class BoundedQueue<T> {
  private items: T[] = [];
  private dropped = 0;

  constructor(private readonly capacity: number) {
    if (capacity <= 0) {
      throw new RangeError("queue capacity must be > 0");
    }
  }

  push(item: T): void {
    if (this.items.length >= this.capacity) {
      this.items.shift();
      this.dropped += 1;
    }
    this.items.push(item);
  }

  drain(max: number): T[] {
    if (max <= 0 || this.items.length === 0) return [];
    return this.items.splice(0, max);
  }

  unshiftAll(items: T[]): void {
    // Used to put items back at the front after a failed flush. We still honour
    // the bound: anything that no longer fits gets counted as dropped.
    const room = this.capacity - this.items.length;
    if (items.length <= room) {
      this.items = items.concat(this.items);
      return;
    }
    const keep = items.slice(items.length - room);
    this.dropped += items.length - keep.length;
    this.items = keep.concat(this.items);
  }

  get size(): number {
    return this.items.length;
  }

  get droppedCount(): number {
    return this.dropped;
  }

  incrementDropped(n: number): void {
    this.dropped += n;
  }
}
