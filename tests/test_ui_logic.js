"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

global.document = { addEventListener() {} };
const source = fs.readFileSync("web/app.js", "utf8");
vm.runInThisContext(source, { filename: "web/app.js" });

const completed = JSON.stringify({
  corrected_segments: [
    { index: 0, corrected_text: "機器學習模型", changes_made: "上下文：學息→學習" },
    { index: 1, corrected_text: "明天開視訊會議", changes_made: "同音字：會意→會議" },
  ],
  summary: "完成",
});

const items = extractCompletedCorrectionItems(completed);
assert.equal(items.length, 2);
assert.equal(items[0].corrected_text, "機器學習模型");
assert.equal(items[1].changes_made, "同音字：會意→會議");

const partial = completed.slice(0, completed.indexOf("},{") + 1);
const partialItems = extractCompletedCorrectionItems(partial);
assert.equal(partialItems.length, 1);

const diff = buildTextDiff("明天開視訊會意", "明天開視訊會議");
assert.match(diff.before, /diff-delete/);
assert.match(diff.after, /diff-add/);

console.log("UI stream parser and text diff tests passed.");
