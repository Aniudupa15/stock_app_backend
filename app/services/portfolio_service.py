import uuid
from datetime import date
from decimal import Decimal

from app.core.exceptions import PortfolioNotFoundError, StockNotFoundError
from app.domain.entities import PortfolioTransaction, TransactionType
from app.domain.ports import MarketMoverRepositoryPort, PortfolioRepositoryPort
from app.finance.holdings import compute_holdings
from app.finance.xirr import xirr
from app.schemas.portfolio import (
    HoldingOut,
    PortfolioDetailOut,
    PortfolioOut,
    PortfolioPerformanceOut,
    TransactionCreate,
)


class PortfolioService:
    def __init__(self, repository: PortfolioRepositoryPort, price_repository: MarketMoverRepositoryPort):
        self._repository = repository
        self._price_repository = price_repository

    async def create(self, user_id: uuid.UUID, name: str) -> PortfolioOut:
        portfolio = await self._repository.create(user_id, name)
        return PortfolioOut(id=portfolio.id, name=portfolio.name, created_at=portfolio.created_at)

    async def list(self, user_id: uuid.UUID) -> list[PortfolioOut]:
        portfolios = await self._repository.list_for_user(user_id)
        return [PortfolioOut(id=p.id, name=p.name, created_at=p.created_at) for p in portfolios]

    async def add_transaction(
        self, user_id: uuid.UUID, portfolio_id: uuid.UUID, body: TransactionCreate
    ) -> PortfolioDetailOut:
        portfolio = await self._repository.get(portfolio_id, user_id)
        if portfolio is None:
            raise PortfolioNotFoundError(portfolio_id)

        transaction = PortfolioTransaction(
            symbol=body.symbol.strip().upper(),
            transaction_type=body.transaction_type,
            quantity=body.quantity,
            price=body.price,
            transaction_date=body.transaction_date,
        )
        added = await self._repository.add_transaction(portfolio_id, transaction)
        if not added:
            raise StockNotFoundError(body.symbol)

        return await self.get_detail(user_id, portfolio_id)

    async def get_detail(self, user_id: uuid.UUID, portfolio_id: uuid.UUID) -> PortfolioDetailOut:
        portfolio = await self._repository.get(portfolio_id, user_id)
        if portfolio is None:
            raise PortfolioNotFoundError(portfolio_id)

        transactions = await self._repository.get_transactions(portfolio_id)
        holdings = compute_holdings(transactions)
        price_by_symbol = {
            p.symbol: p for p in await self._price_repository.get_latest_prices([h.symbol for h in holdings])
        }

        holding_outs = []
        for h in holdings:
            latest = price_by_symbol.get(h.symbol)
            current_price = latest.last_price if latest else None
            current_value = current_price * h.quantity if current_price is not None else None
            pnl = current_value - h.cost_basis if current_value is not None else None
            pnl_percent = (pnl / h.cost_basis * 100) if pnl is not None and h.cost_basis != 0 else None
            holding_outs.append(
                HoldingOut(
                    symbol=h.symbol,
                    quantity=h.quantity,
                    avg_price=h.avg_price,
                    cost_basis=h.cost_basis,
                    current_price=current_price,
                    current_value=current_value,
                    pnl=pnl,
                    pnl_percent=pnl_percent,
                )
            )

        return PortfolioDetailOut(
            id=portfolio.id, name=portfolio.name, created_at=portfolio.created_at, holdings=holding_outs
        )

    async def get_performance(self, user_id: uuid.UUID, portfolio_id: uuid.UUID) -> PortfolioPerformanceOut:
        portfolio = await self._repository.get(portfolio_id, user_id)
        if portfolio is None:
            raise PortfolioNotFoundError(portfolio_id)

        transactions = await self._repository.get_transactions(portfolio_id)
        holdings = compute_holdings(transactions)
        price_by_symbol = {
            p.symbol: p for p in await self._price_repository.get_latest_prices([h.symbol for h in holdings])
        }

        current_value = Decimal(0)
        for h in holdings:
            latest = price_by_symbol.get(h.symbol)
            if latest is not None:
                current_value += latest.last_price * h.quantity

        total_invested = sum(
            (t.quantity * t.price for t in transactions if t.transaction_type == TransactionType.BUY), Decimal(0)
        )
        total_realized = sum(
            (t.quantity * t.price for t in transactions if t.transaction_type == TransactionType.SELL), Decimal(0)
        )
        total_pnl = (current_value + total_realized) - total_invested
        total_pnl_percent = (total_pnl / total_invested * 100) if total_invested != 0 else None

        cash_flows: list[tuple[date, float]] = [
            (
                t.transaction_date,
                float(-(t.quantity * t.price))
                if t.transaction_type == TransactionType.BUY
                else float(t.quantity * t.price),
            )
            for t in transactions
        ]
        if current_value > 0:
            cash_flows.append((date.today(), float(current_value)))

        rate = xirr(cash_flows)

        return PortfolioPerformanceOut(
            id=portfolio.id,
            total_invested=total_invested,
            current_value=current_value,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            xirr_percent=Decimal(str(round(rate * 100, 4))) if rate is not None else None,
        )
