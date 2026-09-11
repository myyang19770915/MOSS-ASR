"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

global.document = { addEventListener() {} };
const source = fs.readFileSync("web/benchmark.js", "utf8");
vm.runInThisContext(source, { filename: "web/benchmark.js" });

const rows = [
  { id: "excellent", score: { accuracy_percent: 98 } },
  { id: "critical", score: { accuracy_percent: 35 } },
  { id: "warning", score: { accuracy_percent: 72 } },
];

assert.deepEqual(reviewRows(rows, true).map((row) => row.id), ["critical", "warning"]);
assert.deepEqual(reviewRows(rows, false).map((row) => row.id), ["critical", "warning", "excellent"]);
assert.deepEqual(reviewRows([{ id: "pass", score: { accuracy_percent: 92 } }], true).map((row) => row.id), ["pass"]);

assert.equal(scoreClass(91), "score-good");
assert.equal(scoreClass(72), "score-warn");
assert.equal(scoreClass(35), "score-critical");

console.log("Benchmark review prioritization tests passed.");
