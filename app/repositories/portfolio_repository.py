import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.entities import Portfolio, PortfolioTransaction
from app.domain.ports import PortfolioRepositoryPort
from app.models.portfolio import PortfolioModel
from app.models.portfolio_transaction import PortfolioTransactionModel
from app.models.stock import StockModel


class SqlAlchemyPortfolioRepository(PortfolioRepositoryPort):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def create(self, user_id: uuid.UUID, name: str) -> Portfolio:
        model = PortfolioModel(user_id=user_id, name=name)
        self._session.add(model)
        await self._session.commit()
        await self._session.refresh(model)
        return Portfolio(id=model.id, user_id=model.user_id, name=model.name, created_at=model.created_at)

    async def list_for_user(self, user_id: uuid.UUID) -> list[Portfolio]:
        stmt = select(PortfolioModel).where(PortfolioModel.user_id == user_id).order_by(PortfolioModel.created_at.asc())
        result = await self._session.execute(stmt)
        return [
            Portfolio(id=row.id, user_id=row.user_id, name=row.name, created_at=row.created_at)
            for row in result.scalars().all()
        ]

    async def get(self, portfolio_id: uuid.UUID, user_id: uuid.UUID) -> Portfolio | None:
        stmt = select(PortfolioModel).where(PortfolioModel.id == portfolio_id, PortfolioModel.user_id == user_id)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is None:
            return None
        return Portfolio(id=row.id, user_id=row.user_id, name=row.name, created_at=row.created_at)

    async def add_transaction(self, portfolio_id: uuid.UUID, transaction: PortfolioTransaction) -> bool:
        stock_id_stmt = select(StockModel.id).where(StockModel.symbol == transaction.symbol.strip().upper())
        stock_id = (await self._session.execute(stock_id_stmt)).scalar_one_or_none()
        if stock_id is None:
            return False

        model = PortfolioTransactionModel(
            portfolio_id=portfolio_id,
            stock_id=stock_id,
            transaction_type=transaction.transaction_type,
            quantity=transaction.quantity,
            price=transaction.price,
            transaction_date=transaction.transaction_date,
        )
        self._session.add(model)
        await self._session.commit()
        return True

    async def get_transactions(self, portfolio_id: uuid.UUID) -> list[PortfolioTransaction]:
        stmt = (
            select(
                StockModel.symbol,
                PortfolioTransactionModel.transaction_type,
                PortfolioTransactionModel.quantity,
                PortfolioTransactionModel.price,
                PortfolioTransactionModel.transaction_date,
            )
            .join(StockModel, StockModel.id == PortfolioTransactionModel.stock_id)
            .where(PortfolioTransactionModel.portfolio_id == portfolio_id)
            .order_by(PortfolioTransactionModel.transaction_date.asc())
        )
        result = await self._session.execute(stmt)
        return [
            PortfolioTransaction(
                symbol=row.symbol,
                transaction_type=row.transaction_type,
                quantity=row.quantity,
                price=row.price,
                transaction_date=row.transaction_date,
            )
            for row in result
        ]
