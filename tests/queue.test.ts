import { describe, expect, it } from "vitest";
import { BoundedQueue } from "../src/queue.js";

describe("BoundedQueue", () => {
  it("rejects non-positive capacity", () => {
    expect(() => new BoundedQueue(0)).toThrow();
    expect(() => new BoundedQueue(-1)).toThrow();
  });

  it("pushes up to capacity without dropping", () => {
    const q = new BoundedQueue<number>(3);
    q.push(1);
    q.push(2);
    q.push(3);
    expect(q.size).toBe(3);
    expect(q.droppedCount).toBe(0);
  });

  it("drops oldest when over capacity", () => {
    const q = new BoundedQueue<number>(3);
    q.push(1);
    q.push(2);
    q.push(3);
    q.push(4);
    expect(q.size).toBe(3);
    expect(q.droppedCount).toBe(1);
    const drained = q.drain(3);
    expect(drained).toEqual([2, 3, 4]);
  });

  it("drain returns at most max items in order", () => {
    const q = new BoundedQueue<number>(10);
    for (let i = 0; i < 5; i++) q.push(i);
    const first = q.drain(2);
    expect(first).toEqual([0, 1]);
    expect(q.size).toBe(3);
    const rest = q.drain(10);
    expect(rest).toEqual([2, 3, 4]);
  });

  it("unshiftAll restores items at the head", () => {
    const q = new BoundedQueue<number>(10);
    q.push(3);
    q.push(4);
    q.unshiftAll([1, 2]);
    const all = q.drain(10);
    expect(all).toEqual([1, 2, 3, 4]);
  });

  it("unshiftAll counts dropped items when over capacity", () => {
    const q = new BoundedQueue<number>(3);
    q.push(10);
    q.push(11);
    q.unshiftAll([1, 2, 3, 4]);
    // Capacity 3 with two existing items leaves room for 1 unshifted item;
    // the rest are counted as dropped.
    expect(q.size).toBe(3);
    expect(q.droppedCount).toBe(3);
    const drained = q.drain(3);
    expect(drained).toEqual([4, 10, 11]);
  });

  it("incrementDropped bumps the counter", () => {
    const q = new BoundedQueue<number>(3);
    q.incrementDropped(5);
    expect(q.droppedCount).toBe(5);
  });
});
