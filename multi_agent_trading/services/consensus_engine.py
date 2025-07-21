"""
Consensus engine for multi-agent decision making.
"""

import logging
from typing import List, Dict, Any
from datetime import datetime
from ..models.trading_models import Vote, TradingProposal


class ConsensusResult:
    """Result of consensus calculation."""
    
    def __init__(self, decision: str, confidence: float, reason: str = ""):
        self.decision = decision
        self.confidence = confidence
        self.reason = reason
        self.timestamp = datetime.utcnow()


class ConsensusEngine:
    """
    Implements weighted voting mechanism for agent consensus.
    """
    
    def __init__(self, threshold: float = 75.0, max_variance: float = 40.0):
        """
        Initialize consensus engine.
        
        Args:
            threshold: Minimum weighted score for execution
            max_variance: Maximum variance before requiring manual review
        """
        self.threshold = threshold
        self.max_variance = max_variance
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_consensus(self, votes: List[Vote]) -> ConsensusResult:
        """
        Calculate consensus from agent votes.
        
        Args:
            votes: List of votes from agents
            
        Returns:
            Consensus result with decision
        """
        if not votes:
            return ConsensusResult("REJECT", 0.0, "No votes received")
        
        # Calculate weighted score
        total_weight = sum(vote.confidence for vote in votes)
        if total_weight == 0:
            return ConsensusResult("REJECT", 0.0, "Zero confidence votes")
        
        weighted_score = sum(vote.score * vote.confidence for vote in votes) / total_weight
        
        # Calculate variance
        scores = [vote.score for vote in votes]
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        self.logger.info(f"Consensus calculation: weighted_score={weighted_score:.2f}, variance={variance:.2f}")
        
        # Make decision
        if variance > self.max_variance:
            return ConsensusResult("MANUAL_REVIEW", weighted_score, f"High disagreement (variance: {variance:.2f})")
        elif weighted_score >= self.threshold:
            return ConsensusResult("EXECUTE", weighted_score, f"Consensus reached (score: {weighted_score:.2f})")
        else:
            return ConsensusResult("REJECT", weighted_score, f"Insufficient consensus (score: {weighted_score:.2f})")
    
    def log_decision(self, proposal: TradingProposal, votes: List[Vote], result: ConsensusResult) -> Dict[str, Any]:
        """
        Log decision for audit trail.
        
        Args:
            proposal: Original trading proposal
            votes: Agent votes
            result: Consensus result
            
        Returns:
            Decision log entry
        """
        decision_log = {
            "proposal_id": proposal.proposal_id,
            "symbol": proposal.symbol,
            "action": proposal.action.value,
            "decision": result.decision,
            "confidence": result.confidence,
            "reason": result.reason,
            "votes": [vote.to_dict() for vote in votes],
            "timestamp": result.timestamp.isoformat()
        }
        
        self.logger.info(f"Decision logged: {decision_log}")
        return decision_log