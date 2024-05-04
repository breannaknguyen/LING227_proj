from __future__ import division

#############################################################################
# Copyright 2011 Jason Baldridge
# 
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#############################################################################

NO_LABEL = "no_label"

class Post:

    def __init__(self):
        pass

    @classmethod
    def fromXmlElement (cls, itemEl):
        """Initialize an AITA Post using an XML element."""
        obj = cls()
        obj.label    = itemEl.get("label", NO_LABEL)
        obj.title    = itemEl.get("title", NO_LABEL)
        obj.op_gender    = itemEl.get("op_gender", NO_LABEL)
        obj.target_gender    = itemEl.get("target_gender", NO_LABEL)
        obj.experience    = itemEl.get("experience", NO_LABEL)
        obj.relationship    = itemEl.get("relationship", NO_LABEL)
        obj.industry    = itemEl.get("industry", NO_LABEL)
        obj.condition    = itemEl.get("condition", NO_LABEL)
        obj.action    = itemEl.get("action", NO_LABEL)
        obj.intention    = itemEl.get("intention", NO_LABEL)
        obj.impact    = itemEl.get("impact", NO_LABEL)
        obj.good_standing    = itemEl.get("good_standing", NO_LABEL)
        obj.perspective    = itemEl.get("perspective", NO_LABEL)
        obj.content  = list(itemEl)[0].text



        return obj

    @classmethod
    def fromValues (cls, label, title, op_gender, target_gender, experience, relationship, industry, condition, action, intention, impact, good_standing, perspective):
        """Initialize a Post from raw values."""
        obj = cls()
        obj.label    = label
        obj.title    = title
        obj.op_gender    = op_gender
        obj.target_gender    = target_gender
        obj.experience    = experience
        obj.relationship    = relationship
        obj.industry    = industry
        obj.condition    = condition
        obj.action    = action
        obj.intention    = intention
        obj.impact    = impact
        obj.good_standing    = good_standing
        obj.perspective    = perspective
        obj.content  = content
        return obj

    def to_string (self):
        return self.title + " : " + self.label + \
            " (" + self.label + ") " + self.content 

    def copy (self):
        return Post.fromValues(self.title, \
                                self.label, self.content)

def convert_posneg_to_subjective (postset):
    for post in postset:
        if post.label == "NTA" or post.label == "YTA":
            post.label = "subjective"
    return postset

def scoreForLabel (eval_labels, gold_labels, label):
    num_correct = sum([e==g and e==label for e,g in zip(eval_labels,gold_labels)])
    num_predicted = eval_labels.count(label)
    num_gold = gold_labels.count(label)
    precision = num_correct/num_predicted * 100.0 if num_predicted != 0 else 0.0
    recall = num_correct/num_gold * 100.0
    fscore = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0.0
    return (precision,recall,fscore,label)

def scorePredictions (eval_posts, gold_posts):
    eval_labels = [x.label for x in eval_posts]
    gold_labels = [x.label for x in gold_posts]
    label_set = set(gold_labels)

    # Get overall accuracy: for each post, was the system correct or incorrect?
    correctness_vector = [e==g for e,g in zip(eval_labels,gold_labels)]
    accuracy = sum(correctness_vector)/len(gold_posts) * 100.0

    # Get label-level results
    results = [scoreForLabel(eval_labels,gold_labels,label) for label in sorted(label_set)]

    return (accuracy, results, correctness_vector)
