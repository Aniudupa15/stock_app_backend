"""Async SQLAlchemy repositories for the 16 trading tables.

Follows the data-service convention: a repository takes an AsyncSession and
commits internally. Config-style rows (accounts, strategies, broker sessions,
risk profiles, backtests) are returned as ORM models; trades/positions/signals
are mapped to their `libs.trading_domain` entities on read. Upserts use
Postgres ON CONFLICT against the schema's unique constraints.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from libs.trading_domain.entities import Position, Signal, Trade
from libs.trading_domain.enums import ExitReason, Product
from services.trading_service.persistence.models import (
    AuditLogModel,
    BacktestModel,
    BracketGroupModel,
    BrokerSessionModel,
    EquitySnapshotModel,
    FillModel,
    HoldingModel,
    OrderIntentModel,
    OrderModel,
    PositionModel,
    RiskEventModel,
    RiskProfileModel,
    SignalModel,
    StrategyModel,
    TradeModel,
    TradingAccountModel,
)


class _Base:
    def __init__(self, session: AsyncSession):
        self._session = session


class TradingAccountRepository(_Base):
    async def create(self, user_id: uuid.UUID, mode: str, starting_balance: Decimal | None) -> TradingAccountModel:
        model = TradingAccountModel(
            user_id=user_id, mode=mode, virtual_balance=starting_balance, starting_balance=starting_balance
        )
        self._session.add(model)
        await self._session.commit()
        await self._session.refresh(model)
        return model

    async def get(self, account_id: uuid.UUID) -> TradingAccountModel | None:
        return await self._session.get(TradingAccountModel, account_id)

    async def list_for_user(self, user_id: uuid.UUID) -> list[TradingAccountModel]:
        stmt = select(TradingAccountModel).where(TradingAccountModel.user_id == user_id)
        return list((await self._session.execute(stmt)).scalars().all())

    async def update_balance(self, account_id: uuid.UUID, virtual_balance: Decimal) -> None:
        await self._session.execute(
            update(TradingAccountModel).where(TradingAccountModel.id == account_id).values(virtual_balance=virtual_balance)
        )
        await self._session.commit()


class BrokerSessionRepository(_Base):
    async def upsert_credentials(
        self, user_id: uuid.UUID, broker: str, api_key_enc: bytes, api_secret_enc: bytes
    ) -> BrokerSessionModel:
        stmt = (
            pg_insert(BrokerSessionModel)
            .values(user_id=user_id, broker=broker, api_key_enc=api_key_enc, api_secret_enc=api_secret_enc)
            .on_conflict_do_update(
                constraint="uq_broker_sessions_user_broker",
                set_={"api_key_enc": api_key_enc, "api_secret_enc": api_secret_enc},
            )
            .returning(BrokerSessionModel.id)
        )
        session_id = (await self._session.execute(stmt)).scalar_one()
        await self._session.commit()
        return await self._session.get(BrokerSessionModel, session_id)

    async def set_access_token(
        self, session_id: uuid.UUID, access_token_enc: bytes, expires_at: datetime, status: str = "CONNECTED"
    ) -> None:
        await self._session.execute(
            update(BrokerSessionModel)
            .where(BrokerSessionModel.id == session_id)
            .values(access_token_enc=access_token_enc, access_token_expires_at=expires_at, status=status)
        )
        await self._session.commit()

    async def set_status(self, session_id: uuid.UUID, status: str) -> None:
        await self._session.execute(
            update(BrokerSessionModel).where(BrokerSessionModel.id == session_id).values(status=status)
        )
        await self._session.commit()

    async def get(self, user_id: uuid.UUID, broker: str) -> BrokerSessionModel | None:
        stmt = select(BrokerSessionModel).where(
            BrokerSessionModel.user_id == user_id, BrokerSessionModel.broker == broker
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()


class StrategyRepository(_Base):
    async def create(self, user_id: uuid.UUID, **fields) -> StrategyModel:
        model = StrategyModel(user_id=user_id, **fields)
        self._session.add(model)
        await self._session.commit()
        await self._session.refresh(model)
        return model

    async def get(self, strategy_id: uuid.UUID, user_id: uuid.UUID) -> StrategyModel | None:
        stmt = select(StrategyModel).where(StrategyModel.id == strategy_id, StrategyModel.user_id == user_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list_for_user(self, user_id: uuid.UUID) -> list[StrategyModel]:
        stmt = select(StrategyModel).where(StrategyModel.user_id == user_id).order_by(StrategyModel.created_at.desc())
        return list((await self._session.execute(stmt)).scalars().all())

    async def update(self, strategy_id: uuid.UUID, user_id: uuid.UUID, **fields) -> bool:
        result = await self._session.execute(
            update(StrategyModel)
            .where(StrategyModel.id == strategy_id, StrategyModel.user_id == user_id)
            .values(**fields)
        )
        await self._session.commit()
        return result.rowcount > 0

    async def delete(self, strategy_id: uuid.UUID, user_id: uuid.UUID) -> bool:
        model = await self.get(strategy_id, user_id)
        if model is None:
            return False
        await self._session.delete(model)
        await self._session.commit()
        return True


class RiskProfileRepository(_Base):
    async def upsert(self, account_id: uuid.UUID, **fields) -> RiskProfileModel:
        stmt = (
            pg_insert(RiskProfileModel)
            .values(account_id=account_id, **fields)
            .on_conflict_do_update(constraint="uq_risk_profiles_account", set_=fields)
            .returning(RiskProfileModel.id)
        )
        profile_id = (await self._session.execute(stmt)).scalar_one()
        await self._session.commit()
        return await self._session.get(RiskProfileModel, profile_id)

    async def get(self, account_id: uuid.UUID) -> RiskProfileModel | None:
        stmt = select(RiskProfileModel).where(RiskProfileModel.account_id == account_id)
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def set_kill_switch(self, account_id: uuid.UUID, on: bool) -> None:
        await self._session.execute(
            update(RiskProfileModel).where(RiskProfileModel.account_id == account_id).values(kill_switch=on)
        )
        await self._session.commit()


class OrderIntentRepository(_Base):
    async def save(self, intent, account_id: uuid.UUID, risk_verdict: dict | None = None) -> uuid.UUID:
        model = OrderIntentModel(
            id=intent.intent_id,
            account_id=account_id,
            strategy_id=intent.strategy_id,
            signal_id=intent.signal_id,
            symbol=intent.symbol,
            side=intent.side.value,
            order_type=intent.order_type.value,
            product=intent.product.value,
            quantity=intent.quantity,
            price=intent.price,
            trigger_price=intent.trigger_price,
            validity=intent.validity.value,
            risk_verdict=risk_verdict,
        )
        self._session.add(model)
        await self._session.commit()
        return model.id


class OrderRepository(_Base):
    async def create(
        self,
        intent_id: uuid.UUID,
        account_id: uuid.UUID,
        venue: str,
        venue_order_id: str | None,
        state: str,
        leg: str | None = None,
        parent_order_id: uuid.UUID | None = None,
    ) -> OrderModel:
        model = OrderModel(
            intent_id=intent_id,
            account_id=account_id,
            venue=venue,
            venue_order_id=venue_order_id,
            state=state,
            leg=leg,
            parent_order_id=parent_order_id,
        )
        self._session.add(model)
        await self._session.commit()
        await self._session.refresh(model)
        return model

    async def update_from_event(
        self, venue: str, venue_order_id: str, state: str, filled_qty: int, pending_qty: int, average_price: Decimal | None
    ) -> None:
        await self._session.execute(
            update(OrderModel)
            .where(OrderModel.venue == venue, OrderModel.venue_order_id == venue_order_id)
            .values(state=state, filled_qty=filled_qty, pending_qty=pending_qty, average_price=average_price)
        )
        await self._session.commit()

    async def list_for_account(self, account_id: uuid.UUID) -> list[OrderModel]:
        stmt = select(OrderModel).where(OrderModel.account_id == account_id).order_by(OrderModel.created_at.desc())
        return list((await self._session.execute(stmt)).scalars().all())


class FillRepository(_Base):
    async def add(
        self, order_id: uuid.UUID, symbol: str, side: str, qty: int, price: Decimal, charges: dict | None, ts: datetime | None = None
    ) -> None:
        model = FillModel(order_id=order_id, symbol=symbol, side=side, qty=qty, price=price, charges=charges)
        if ts is not None:
            model.ts = ts
        self._session.add(model)
        await self._session.commit()


class TradeRepository(_Base):
    async def add(self, trade: Trade) -> uuid.UUID:
        model = TradeModel(
            account_id=trade.account_id,
            strategy_id=trade.strategy_id,
            symbol=trade.symbol,
            entry_ts=trade.entry_ts,
            exit_ts=trade.exit_ts,
            qty=trade.qty,
            entry_price=trade.entry_price,
            exit_price=trade.exit_price,
            pnl_gross=trade.pnl_gross,
            charges_total=trade.charges_total,
            pnl_net=trade.pnl_net,
            r_multiple=None,  # not carried on the domain Trade; computed/enriched later
            holding_seconds=trade.holding_seconds,
            exit_reason=trade.exit_reason.value if trade.exit_reason else None,
        )
        self._session.add(model)
        await self._session.commit()
        return model.id

    async def list_for_account(self, account_id: uuid.UUID) -> list[Trade]:
        stmt = select(TradeModel).where(TradeModel.account_id == account_id).order_by(TradeModel.exit_ts.asc())
        rows = (await self._session.execute(stmt)).scalars().all()
        return [
            Trade(
                account_id=r.account_id,
                symbol=r.symbol,
                qty=r.qty,
                entry_price=r.entry_price,
                exit_price=r.exit_price,
                pnl_gross=r.pnl_gross,
                charges_total=r.charges_total,
                pnl_net=r.pnl_net,
                entry_ts=r.entry_ts,
                exit_ts=r.exit_ts,
                strategy_id=r.strategy_id,
                exit_reason=ExitReason(r.exit_reason) if r.exit_reason else None,
            )
            for r in rows
        ]


class PositionRepository(_Base):
    async def upsert(self, account_id: uuid.UUID, symbol: str, product: str, net_qty: int, avg_price: Decimal, realized_pnl: Decimal) -> None:
        stmt = (
            pg_insert(PositionModel)
            .values(
                account_id=account_id, symbol=symbol, product=product, net_qty=net_qty, avg_price=avg_price, realized_pnl=realized_pnl
            )
            .on_conflict_do_update(
                constraint="uq_positions_account_symbol_product",
                set_={"net_qty": net_qty, "avg_price": avg_price, "realized_pnl": realized_pnl},
            )
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def list_for_account(self, account_id: uuid.UUID) -> list[Position]:
        stmt = select(PositionModel).where(PositionModel.account_id == account_id)
        rows = (await self._session.execute(stmt)).scalars().all()
        return [
            Position(
                account_id=r.account_id,
                symbol=r.symbol,
                product=Product(r.product),
                net_qty=r.net_qty,
                avg_price=r.avg_price,
                realized_pnl=r.realized_pnl,
            )
            for r in rows
        ]


class HoldingRepository(_Base):
    async def upsert(self, account_id: uuid.UUID, symbol: str, quantity: int, avg_price: Decimal, ltp: Decimal | None) -> None:
        stmt = (
            pg_insert(HoldingModel)
            .values(account_id=account_id, symbol=symbol, quantity=quantity, avg_price=avg_price, ltp=ltp)
            .on_conflict_do_update(
                constraint="uq_holdings_account_symbol", set_={"quantity": quantity, "avg_price": avg_price, "ltp": ltp}
            )
        )
        await self._session.execute(stmt)
        await self._session.commit()

    async def list_for_account(self, account_id: uuid.UUID) -> list[HoldingModel]:
        stmt = select(HoldingModel).where(HoldingModel.account_id == account_id)
        return list((await self._session.execute(stmt)).scalars().all())


class EquitySnapshotRepository(_Base):
    async def append(self, account_id: uuid.UUID, equity: Decimal, cash: Decimal, unrealized: Decimal | None = None, ts: datetime | None = None) -> None:
        model = EquitySnapshotModel(account_id=account_id, equity=equity, cash=cash, unrealized=unrealized)
        if ts is not None:
            model.ts = ts
        self._session.add(model)
        await self._session.commit()

    async def curve(self, account_id: uuid.UUID) -> list[EquitySnapshotModel]:
        stmt = select(EquitySnapshotModel).where(EquitySnapshotModel.account_id == account_id).order_by(EquitySnapshotModel.ts.asc())
        return list((await self._session.execute(stmt)).scalars().all())


class SignalRepository(_Base):
    async def add(self, signal: Signal, account_id: uuid.UUID | None = None) -> uuid.UUID:
        model = SignalModel(
            strategy_id=signal.strategy_id,
            account_id=account_id,
            symbol=signal.symbol,
            side=signal.side.value,
            entry=signal.entry,
            stop_loss=signal.stop_loss,
            targets=[str(t) for t in signal.targets],
            confidence=signal.confidence,
            reasoning=signal.reasoning,
        )
        self._session.add(model)
        await self._session.commit()
        return model.id

    async def list_for_account(self, account_id: uuid.UUID, limit: int = 100) -> list[SignalModel]:
        stmt = (
            select(SignalModel).where(SignalModel.account_id == account_id).order_by(SignalModel.ts.desc()).limit(limit)
        )
        return list((await self._session.execute(stmt)).scalars().all())


class RiskEventRepository(_Base):
    async def append(self, account_id: uuid.UUID, kind: str, detail: dict | None = None) -> None:
        self._session.add(RiskEventModel(account_id=account_id, kind=kind, detail=detail))
        await self._session.commit()


class AuditLogRepository(_Base):
    async def append(
        self, actor: str, event_type: str, account_id: uuid.UUID | None = None, ref_id: uuid.UUID | None = None, payload: dict | None = None
    ) -> None:
        self._session.add(
            AuditLogModel(account_id=account_id, actor=actor, event_type=event_type, ref_id=ref_id, payload=payload)
        )
        await self._session.commit()

    async def list_for_account(self, account_id: uuid.UUID, limit: int = 200) -> list[AuditLogModel]:
        stmt = select(AuditLogModel).where(AuditLogModel.account_id == account_id).order_by(AuditLogModel.ts.desc()).limit(limit)
        return list((await self._session.execute(stmt)).scalars().all())


class BacktestRepository(_Base):
    async def save(
        self,
        symbol: str,
        starting_cash: Decimal,
        final_equity: Decimal,
        metrics: dict,
        strategy_id: uuid.UUID | None = None,
        account_id: uuid.UUID | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
    ) -> uuid.UUID:
        model = BacktestModel(
            account_id=account_id,
            strategy_id=strategy_id,
            symbol=symbol,
            from_date=from_date,
            to_date=to_date,
            starting_cash=starting_cash,
            final_equity=final_equity,
            metrics=metrics,
        )
        self._session.add(model)
        await self._session.commit()
        return model.id

    async def list_recent(self, limit: int = 50) -> list[BacktestModel]:
        stmt = select(BacktestModel).order_by(BacktestModel.created_at.desc()).limit(limit)
        return list((await self._session.execute(stmt)).scalars().all())


class BracketGroupRepository(_Base):
    async def create(
        self,
        account_id: uuid.UUID,
        entry_order_id: uuid.UUID | None = None,
        sl_order_id: uuid.UUID | None = None,
        target_order_ids: list | None = None,
        trailing: dict | None = None,
    ) -> BracketGroupModel:
        model = BracketGroupModel(
            account_id=account_id,
            entry_order_id=entry_order_id,
            sl_order_id=sl_order_id,
            target_order_ids=target_order_ids,
            trailing=trailing,
        )
        self._session.add(model)
        await self._session.commit()
        await self._session.refresh(model)
        return model

    async def set_state(self, bracket_id: uuid.UUID, state: str) -> None:
        await self._session.execute(
            update(BracketGroupModel).where(BracketGroupModel.id == bracket_id).values(state=state)
        )
        await self._session.commit()
