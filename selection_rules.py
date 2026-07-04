from experta import *
from facts import *

class SelectionRules(KnowledgeEngine):

    @DefFacts()
    def initial_facts(self):
        yield SelectedArchitecture(
            name="None",
            reason="No architecture selected yet.",
            source_rule="INIT"
        )

    # =====================================================
    # SR_00A : إنشاء مرشح جديد من دليل النقاط
    # =====================================================
    @Rule(
        AS.ev << ScoreEvidence(arch=MATCH.n, points=MATCH.p, reason=MATCH.r, source_rule=MATCH.sr),
        NOT(CandidateArchitecture(name=MATCH.n))
    )
    def create_candidate_from_evidence(self, ev, n, p, r, sr):
        self.retract(ev)
        # تم إضافة source_rule هنا لكي يتطابق مع القاعدة النهائية
        self.declare(CandidateArchitecture(name=n, score=p, reason=r, source_rule=sr))

    # =====================================================
    # SR_00B : دمج النقاط إذا كانت المعمارية موجودة مسبقاً
    # =====================================================
    @Rule(
        AS.ev << ScoreEvidence(arch=MATCH.n, points=MATCH.p, reason=MATCH.new_r, source_rule=MATCH.new_sr),
        AS.cand << CandidateArchitecture(name=MATCH.n, score=MATCH.s, reason=MATCH.old_r, source_rule=MATCH.old_sr)
    )
    def accumulate_candidate_score(self, ev, cand, n, p, s, old_r, new_r, old_sr, new_sr):
        self.retract(ev)
        combined_reason = old_r + " | " + new_r
        combined_rule = old_sr + " & " + new_sr
        self.modify(cand, score=s + p, reason=combined_reason, source_rule=combined_rule)

    # =====================================================
    # SR_01 : حذف المعمارية ذات النقاط الأقل
    # =====================================================
    @Rule(
        AS.lower_candidate << CandidateArchitecture(score=MATCH.s1),
        CandidateArchitecture(score=MATCH.s2),
        TEST(lambda s1, s2: s1 < s2)
    )
    def eliminate_lower_score(self, lower_candidate):
        self.retract(lower_candidate)

    # =====================================================
    # SR_02 : كسر التعادل (Tie-breaker)
    # =====================================================
    @Rule(
        AS.c1 << CandidateArchitecture(name=MATCH.n1, score=MATCH.s),
        AS.c2 << CandidateArchitecture(name=MATCH.n2, score=MATCH.s),
        TEST(lambda n1, n2: n1 > n2) 
    )
    def tie_breaker(self, c1):
        self.retract(c1)

    # =====================================================
    # SR_03 : تحديد المعمارية الفائزة النهائية
    # =====================================================
    @Rule(
        AS.best << SelectedArchitecture(name="None"),
        CandidateArchitecture(name=MATCH.name, score=MATCH.score, reason=MATCH.reason, source_rule=MATCH.rule),
        salience=-10
    )
    def finalize_best_architecture(self, best, name, score, reason, rule):
        self.modify(
            best,
            name=name,
            reason=reason,
            source_rule=rule
        )